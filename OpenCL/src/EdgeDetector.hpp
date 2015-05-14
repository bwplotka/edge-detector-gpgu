#ifndef Edge_Detector_H_
#define Edge_Detector_H_
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <string.h>
#include "OpenCLUtil.hpp"
#include "SDKBitMap.hpp"

using namespace appsdk;

#define GAUSSIAN_KERNEL "kernels/Gaussian_Kernels.cl"
#define GREYSCALE_KERNEL "kernels/GreyScale_Kernels.cl"
#define MAX_KERNEL "kernels/Max_Kernels.cl"
#define HYSTERESIS_KERNEL "kernels/Hysteresis_Kernels.cl"
#define SOBEL_FILTER_KERNEL "kernels/SobelFilter_Kernels.cl"


#define INPUT_IMAGE "Input_Image.bmp"
#define OUTPUT_IMAGE "Output_Image1.bmp"

#define GROUP_SIZE 8

/**
* EdgeDetector
* Class implements OpenCL Sobel Filter sample
*/

class EdgeDetector
{
        cl_double setupTime;                /**< time taken to setup OpenCL resources and building kernel */
        cl_double kernelTime;               /**< time taken to run kernel and read result back */
        cl_uchar4* inputImageData;          /**< Input bitmap data to device */
        cl_uchar4* outputImageData;         /**< Output from device */
		//cl_uchar4* nextImageData;
        cl_context context;                 /**< CL context */
        cl_device_id *devices;              /**< CL device list */
        cl_mem inputImageBuffer;            /**< CL memory buffer for input Image*/
        cl_mem prevImageBuffer;           /**< CL memory buffer for Output Image*/
		cl_mem nextImageBuffer;
		//cl_mem buffers_[2];
		cl_mem thetaBuffer;
        cl_uchar* verificationOutput;       /**< Output array for reference implementation */
        cl_command_queue commandQueue;      /**< CL command queue */
        cl_program programGrey;                 /**< CL program  */
		cl_program programGaus;
		cl_program programSobel;
		cl_program programMax;
		cl_program programHyst;
        cl_kernel kernelGrey;                   /**< CL kernel */
		cl_kernel kernelGaus;
		cl_kernel kernelSobel;
		cl_kernel kernelMax;
		cl_kernel kernelHyst;
		//cl_kernel kernel;
        SDKBitMap inputBitmap;   /**< Bitmap class object */
        uchar4* pixelData;       /**< Pointer to image data */
        cl_uint pixelSize;                  /**< Size of a pixel in BMP format> */
        cl_uint width;                      /**< Width of image */
        cl_uint height;                     /**< Height of image */
		cl_uint width_original;
		cl_uint height_original;
        cl_bool byteRWSupport;
        size_t kernelWorkGroupSize;         /**< Group Size returned by kernel */
        size_t blockSizeX;                  /**< Work-group size in x-direction */
        size_t blockSizeY;                  /**< Work-group size in y-direction */
		size_t buffer_index_ = 0;
        int iterations;                     /**< Number of iterations for kernel execution */
        SDKDeviceInfo
        deviceInfo;                       /**< Structure to store device information*/
        KernelWorkGroupInfo
        kernelInfo;         /**< Structure to store kernel related info */

        SDKTimer    *sampleTimer;      /**< SDKTimer object */


    public:

		CLContext   *sdkContext;   /**< CLCommand argument class */

        /**
        * Read bitmap image and allocate host memory
        * @param inputImageName name of the input file
        * @return SDK_SUCCESS on success and SDK_FAILURE on failure
        */
        int readInputImage(std::string inputImageName);

        /**
        * Write to an image file
        * @param outputImageName name of the output file
        * @return SDK_SUCCESS on success and SDK_FAILURE on failure
        */
        int writeOutputImage(std::string outputImageName);

        /**
        * Constructor
        * Initialize member variables
        */
        EdgeDetector()
            : inputImageData(NULL),
              outputImageData(NULL),
			  //nextImageData(NULL),
              verificationOutput(NULL),
              byteRWSupport(true)
        {
            sdkContext = new CLContext();
            sampleTimer = new SDKTimer();
            pixelSize = sizeof(uchar4);
            pixelData = NULL;
            blockSizeX = GROUP_SIZE;
            blockSizeY = 1;
            iterations = 1;
			
        }

        ~EdgeDetector()
        {
        }

        /**
        * Allocate image memory and Load bitmap file
        * @return SDK_SUCCESS on success and SDK_FAILURE on failure
        */
        int setupEdgeDetector();

        /**
        * OpenCL related initialisations.
        * Set up Context, Device list, Command Queue, Memory buffers
        * Build CL kernel program executable
        * @return SDK_SUCCESS on success and SDK_FAILURE on failure
        */
        int setupCL();

        /**
        * Set values for kernels' arguments, enqueue calls to the kernels
        * on to the command queue, wait till end of kernel execution.
        * Get kernel start and end time if timing is enabled
        * @return SDK_SUCCESS on success and SDK_FAILURE on failure
        */
        int runCLKernels();

        /**
        * Reference CPU implementation of Binomial Option
        * for performance comparison
        */
        void EdgeDetectorCPUReference();

        /**
        * Override from SDKSample. Print sample stats.
        */
        void printStats();

        /**
        * Override from SDKSample. Initialize
        * command line parser, add custom options
        * @return SDK_SUCCESS on success and SDK_FAILURE on failure
        */
        int initialize();

        /**
         * Override from SDKSample, Generate binary image of given kernel
         * and exit application
        * @return SDK_SUCCESS on success and SDK_FAILURE on failure
         */
        int genBinaryImage();

        /**
        * Override from SDKSample, adjust width and height
        * of execution domain, perform all sample setup
        * @return SDK_SUCCESS on success and SDK_FAILURE on failure
        */
        int setup();

        /**
        * Override from SDKSample
        * Run OpenCL Sobel Filter
        * @return SDK_SUCCESS on success and SDK_FAILURE on failure
        */
        int run();

        /**
        * Override from SDKSample
        * Cleanup memory allocations
        * @return SDK_SUCCESS on success and SDK_FAILURE on failure
        */
        int cleanup();

        /**
        * Override from SDKSample
        * Verify against reference implementation
        * @return SDK_SUCCESS on success and SDK_FAILURE on failure
        */
        int verifyResults();



		int GreyScale();

		int Gaussian();

		int Sobel();

		int Hysteresis();

		int Max();

		//inline cl_mem& NextBuff() { return buffers_[buffer_index_]; }

		//inline cl_mem& PrevBuff() { return buffers_[buffer_index_ ^ 1]; }

		//inline void AdvanceBuff() { buffer_index_ ^= 1; }
};

#endif // Edge_Detector_H_
