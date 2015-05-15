#include <stdio.h>
#include "mpi.h"
#include <math.h>
#include <stdlib.h>
#include "EdgeDetector.hpp"

#define LOG1(X,Y) {printf(X,Y); fflush(stdout); }
#define LOG(X) {printf(X); fflush(stdout); }
#define COMPUTE_TAG 0
#define FINISH_TAG 1

#define MPI_GROUP_SIZE 256

#define DEBUG

#ifdef DEBUG 
#define DBGLOG(X) printf(X); fflush(stdout)
#define DBGLOG1(X, Y) printf(X,Y); fflush(stdout)
#define DBGLOG2(X, Y, Y1) printf(X,Y,Y1); fflush(stdout) 
#define DBGLOG5(X, Y, Y2, Y3, Y4, Y5) printf(X,Y, Y2, Y3, Y4, Y5); fflush(stdout)
#else
#define DBGLOG(X, Y) {}
#endif

struct WorkChunk {
	uchar4 px[MPI_GROUP_SIZE*MPI_GROUP_SIZE];
	int id;
};

//typedef struct
//{
//	unsigned char x;
//	unsigned char y;
//	unsigned char z;
//	unsigned char w;
//} uchar4;

int
readInputImage(SDKBitMap* inputBitmap, std::string inputImageName)
{
	// load input bitmap image
	inputBitmap->load(inputImageName.c_str());

	// error if image did not load
	if (!inputBitmap->isLoaded())
	{
		LOG("Failed to load input image!\n")
			return SDK_FAILURE;
	}

	return SDK_SUCCESS;
}


int
writeOutputImage(SDKBitMap* inputBitmap, std::string outputImageName)
{
	if (!inputBitmap->write(outputImageName.c_str()))
	{
		LOG("\nFailed to write output image!\n");
		return SDK_FAILURE;
	}

	return SDK_SUCCESS;
}

int compute_edges(EdgeDetector clEdgeDetector, WorkChunk* work_chunk){
	cl_int status = 0;

	if (clEdgeDetector.sdkContext->isDumpBinaryEnabled())
	{
		return clEdgeDetector.genBinaryImage();
	}

	status = clEdgeDetector.readInputImage(work_chunk->px, MPI_GROUP_SIZE, MPI_GROUP_SIZE);
	CHECK_ERROR(status, SDK_SUCCESS, "Read InputImage failed\n");

	status = clEdgeDetector.setup();
	if (status != SDK_SUCCESS)
	{
		return status;
	}

	if (clEdgeDetector.run() != SDK_SUCCESS)
	{
		return SDK_FAILURE;
	}

	// write the output image to bitmap file
	status = clEdgeDetector.writeOutputImage(work_chunk->px);
	CHECK_ERROR(status, SDK_SUCCESS, "write Output Image Failed\n");

	return SDK_SUCCESS;
}


int run_process(int my_id, int all_processes) {

	MPI_Request* requests;
	WorkChunk* outputChunks;
	WorkChunk* chunk;
	uchar4* pixelData;
	cl_int status = 0;
	cl_uint width;                      /**< Width of image */
	cl_uint height;                     /**< Height of image */
	cl_uint width_original;
	cl_uint height_original;
	cl_uint height_parts;
	cl_uint width_parts;
	SDKBitMap inputImage;
	//CUSTOM MPI DATATYPES--------
	//construct uchar4 Type:
	const int nitems = 4;
	int          blocklengths[4] = { 1, 1, 1, 1 };
	MPI_Datatype types[4] = { MPI_UNSIGNED_CHAR, MPI_UNSIGNED_CHAR, MPI_UNSIGNED_CHAR, MPI_UNSIGNED_CHAR };
	MPI_Datatype mpi_uchar4_type;
	MPI_Aint     offsets[4];

	offsets[0] = offsetof(uchar4, x);
	offsets[1] = offsetof(uchar4, y);
	offsets[2] = offsetof(uchar4, z);
	offsets[3] = offsetof(uchar4, w);
	MPI_Type_create_struct(nitems, blocklengths, offsets, types, &mpi_uchar4_type);
	MPI_Type_commit(&mpi_uchar4_type);
	//construct WorkChunk Type:
	const int nitems2 = 2;
	int          blocklengths2[2] = { MPI_GROUP_SIZE*MPI_GROUP_SIZE, 1 };
	MPI_Datatype types2[2] = { mpi_uchar4_type, MPI_INT };
	MPI_Datatype mpi_workchunk_type;
	MPI_Aint     offsets2[2];

	offsets2[0] = offsetof(WorkChunk, px);
	offsets2[1] = offsetof(WorkChunk, id);

	MPI_Type_create_struct(nitems2, blocklengths2, offsets2, types2, &mpi_workchunk_type);
	MPI_Type_commit(&mpi_workchunk_type);
	//END CUSTOM MPI DATATYPES--------
	if (!my_id){
		//Master code
		requests = (MPI_Request *)malloc((all_processes - 1)*sizeof(MPI_Request));
		if (!requests) {
			printf("\nNot enough memory");
			MPI_Finalize();
			return -1;
		}
		outputChunks = (WorkChunk *)malloc((all_processes - 1)*sizeof(WorkChunk));
		if (!outputChunks) {
			printf("\nNot enough memory");
			MPI_Finalize();
			return -1;
		}
		int chunk_id = 0;
		int chunks_todo;
		
		std::string filePath = getPath() + std::string(INPUT_IMAGE);
		DBGLOG1("Input File: %s\n", filePath.c_str());
		if (readInputImage(&inputImage, INPUT_IMAGE) == SDK_FAILURE){
			return SDK_FAILURE;
		}
		//DO REMOVE it - don't know why but it helps!
		writeOutputImage(&inputImage, "tmp");
		
		// get width and height of input image
		height_original = inputImage.getHeight();
		width_original = inputImage.getWidth();
		//For now - static group size - TODO: Calculate proper group size dynamically!
		height_parts = ((height_original) / MPI_GROUP_SIZE);
		height = height_parts * MPI_GROUP_SIZE;
		width_parts = ((width_original) / MPI_GROUP_SIZE);
		width = width_parts * MPI_GROUP_SIZE;
		chunks_todo = width_parts * height_parts;
		DBGLOG5("Loaded image h(%d), w(%d) - w parts (%d), h parts (%d), chunks (%d)\n", height, width, height_parts, width_parts, chunks_todo);
		pixelData = inputImage.getPixels();
		if (pixelData == NULL)
		{
			std::cout << "Failed to read pixel Data!\n";
			return SDK_FAILURE;
		}
		int checkpointed_chunks = 0;
		//Main loop - on each iteration give some chunks to 
		while (chunk_id < chunks_todo){
			for (int i = 1; i < all_processes && chunk_id < chunks_todo; chunk_id++, i++){
				WorkChunk todo_chunk;
				todo_chunk.id = chunk_id;
				for (int x = 0; x < MPI_GROUP_SIZE*MPI_GROUP_SIZE; x++){
					//Need to fine tune - one pixel stays
					int yA = ((chunk_id / width_parts)*MPI_GROUP_SIZE) + (x / MPI_GROUP_SIZE);
					int xA = ((chunk_id % width_parts)*MPI_GROUP_SIZE) + (x % MPI_GROUP_SIZE);
					//DBGLOG2("(%d , %d) \n ", xA, yA);
					todo_chunk.px[x] = pixelData[(yA*width) + xA];
				}

				MPI_Send(&todo_chunk, 1, mpi_workchunk_type, i, COMPUTE_TAG, MPI_COMM_WORLD);
				MPI_Irecv(&outputChunks[i-1], 1, mpi_workchunk_type, i, COMPUTE_TAG, MPI_COMM_WORLD, &requests[i - 1]);
				DBGLOG2("\nSend todo chunk (id = %d/%d)", chunk_id, chunks_todo);
			}
			
			MPI_Waitall(chunk_id - checkpointed_chunks, requests, MPI_STATUSES_IGNORE);
			for (int i = 1; i <= chunk_id - checkpointed_chunks; i++){
				DBGLOG2("\nGot work output from process %d - chunk %d", i, outputChunks[i-1].id);
				for (int x = 0; x < MPI_GROUP_SIZE*MPI_GROUP_SIZE; x++){
					int yA = ((outputChunks[i - 1].id / width_parts)*MPI_GROUP_SIZE) + (x / MPI_GROUP_SIZE);
					int xA = ((outputChunks[i - 1].id % width_parts)*MPI_GROUP_SIZE) + (x % MPI_GROUP_SIZE);
					pixelData[(yA*width) + xA] = outputChunks[i - 1].px[x];
				}
			}
			checkpointed_chunks = chunk_id;
		}
	
		LOG("\nFinishing! - Merging image and saving!")
		if (writeOutputImage(&inputImage, OUTPUT_IMAGE) != SDK_SUCCESS){
			LOG("\nError")
		}
		else
			LOG("\nImage saved.")
		for (int i = 1; i < all_processes; i++){
			MPI_Send(0, 0, mpi_workchunk_type, i, FINISH_TAG, MPI_COMM_WORLD);
		}
		FREE(requests);
		return SDK_SUCCESS;
	}

	//Slave code
	chunk = (WorkChunk *)malloc(sizeof(WorkChunk));
	if (!chunk) {
		printf("\nNot enough memory");
		MPI_Finalize();
		return SDK_FAILURE;
	}

	EdgeDetector clEdgeDetector;

	if (clEdgeDetector.initialize() != SDK_SUCCESS)
	{
		LOG("Error")
		return SDK_FAILURE;
	}
	char **argv = (char * *)malloc(sizeof(char *));
	argv[0] = (char *)malloc(sizeof(char));
	argv[0] = "2";
	if (clEdgeDetector.sdkContext->parseCommandLine(1, argv) != SDK_SUCCESS)
	{
		LOG("Error")
		return SDK_FAILURE;
	}

	MPI_Status r_status;
	while (true){
		MPI_Recv(chunk, 1, mpi_workchunk_type, 0, MPI_ANY_TAG, MPI_COMM_WORLD, &r_status);
		if (r_status.MPI_TAG == FINISH_TAG) break;
		DBGLOG2("W %d, got chunk id= %d", my_id, chunk->id);
		//TODO fine tune edges!
		if (compute_edges(clEdgeDetector, chunk) == SDK_FAILURE){
			LOG1("W: %d - error during edge computing!", my_id)
		}
		chunk->px[2].x = 44;
		MPI_Send(chunk, 1, mpi_workchunk_type, 0, COMPUTE_TAG, MPI_COMM_WORLD);
	}
	/*if (clEdgeDetector.verifyResults() != SDK_SUCCESS)
	{
	return SDK_FAILURE;
	}*/
	LOG1("\nW: %d - Finished!\n",my_id);
	//Impossible to clean up - TODO - check it!

	/*if (clEdgeDetector.cleanup() != SDK_SUCCESS)
	{
		return SDK_FAILURE;
	}*/
	/*FREE(chunk)
		FREE(argv[0])
	FREE(argv)*/
	return SDK_SUCCESS;
}
