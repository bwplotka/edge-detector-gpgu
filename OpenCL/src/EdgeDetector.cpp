#include "EdgeDetector.hpp"
#include <cmath>


int
EdgeDetector::readInputImage(std::string inputImageName)
{

    // load input bitmap image
    inputBitmap.load(inputImageName.c_str());

    // error if image did not load
    if(!inputBitmap.isLoaded())
    {
        std::cout << "Failed to load input image!";
        return SDK_FAILURE;
    }


    // get width and height of input image
    height_original = inputBitmap.getHeight();
    width_original = inputBitmap.getWidth();
	height = ((height_original) / GROUP_SIZE) * GROUP_SIZE ;
	width = ((width_original) / GROUP_SIZE) * GROUP_SIZE ;
    // allocate memory for input & output image data
	inputImageData = (cl_uchar4*)malloc(width_original * height_original * sizeof(cl_uchar4));
    CHECK_ALLOCATION(inputImageData, "Failed to allocate memory! (inputImageData)");

	outputImageData = (cl_uchar4*)malloc(width_original * height_original * sizeof(cl_uchar4));

	pixelData = inputBitmap.getPixels();
    if(pixelData == NULL)
    {
        std::cout << "Failed to read pixel Data!";
        return SDK_FAILURE;
    }

    // Copy pixel data into inputImageData
	memcpy(inputImageData, pixelData, width_original * height_original * pixelSize);
    // allocate memory for verification output
	verificationOutput = (cl_uchar*)malloc(width_original * height_original * pixelSize);
    CHECK_ALLOCATION(verificationOutput,
                     "verificationOutput heap allocation failed!");

    // initialize the data to NULL
	memset(verificationOutput, 0, width_original * height_original * pixelSize);

    return SDK_SUCCESS;

}


int
EdgeDetector::writeOutputImage(std::string outputImageName)
{
    // copy output image data back to original pixel data
	memcpy(pixelData, outputImageData,
		width_original * height_original * pixelSize);

	//inputBitmap.height = height;
	//inputBitmap.width = width;
    // write the output bmp file
    if(!inputBitmap.write(outputImageName.c_str()))
    {
        std::cout << "Failed to write output image!";
        return SDK_FAILURE;
    }

    return SDK_SUCCESS;
}

int
EdgeDetector::genBinaryImage()
{
    bifData binaryData;
    binaryData.kernelName = std::string(SOBEL_FILTER_KERNEL);
    binaryData.flagsStr = std::string("");
    if(sdkContext->isComplierFlagsSpecified())
    {
        binaryData.flagsFileName = std::string(sdkContext->flags.c_str());
    }

    binaryData.binaryName = std::string(sdkContext->dumpBinary.c_str());
	int status = generateBinaryImage(binaryData, sdkContext);
    return status;
}


int
EdgeDetector::setupCL()
{
    cl_int status = CL_SUCCESS;
    cl_device_type dType;

    if(sdkContext->deviceType.compare("cpu") == 0)
    {
        dType = CL_DEVICE_TYPE_CPU;
    }
    else //deviceType = "gpu"
    {
        dType = CL_DEVICE_TYPE_GPU;
        if(sdkContext->isThereGPU() == false)
        {
            std::cout << "GPU not found. Falling back to CPU device" << std::endl;
            dType = CL_DEVICE_TYPE_CPU;
        }
    }

	cl_platform_id platform = sdkContext->platforms[sdkContext->chosen_platform];

    // Display available devices.
	int retValue = displayDevices(platform, dType);
    CHECK_ERROR(retValue, SDK_SUCCESS, "displayDevices() failed");
	std::cout << "Selected Device: " << sdkContext->deviceId  << std::endl;
	
    // If we could find our platform, use it. Otherwise use just available platform.
    cl_context_properties cps[3] =
    {
        CL_CONTEXT_PLATFORM,
        (cl_context_properties)platform,
        0
    };

    context = clCreateContextFromType(
                  cps,
                  dType,
                  NULL,
                  NULL,
                  &status);
    CHECK_OPENCL_ERROR( status, "clCreateContextFromType failed.");

    // getting device on which to run the sample
    status = getDevices(context,&devices,sdkContext->deviceId,
                        sdkContext->isDeviceIdEnabled());
    CHECK_ERROR(status, SDK_SUCCESS, "getDevices() failed");

    {
        // The block is to move the declaration of prop closer to its use
        cl_command_queue_properties prop = 0;
        commandQueue = clCreateCommandQueue(
                           context,
                           devices[sdkContext->deviceId],
                           prop,
                           &status);
        CHECK_OPENCL_ERROR( status, "clCreateCommandQueue failed.");
    }

    //Set device info of given cl_device_id
    retValue = deviceInfo.setDeviceInfo(devices[sdkContext->deviceId]);
    CHECK_ERROR(retValue, 0, "SDKDeviceInfo::setDeviceInfo() failed");


    // Create and initialize memory objects

    // Set Presistent memory only for AMD platform
    cl_mem_flags inMemFlags = CL_MEM_READ_ONLY;
    

    // Create memory object for input Image
    inputImageBuffer = clCreateBuffer(
                           context,
                           inMemFlags,
						   width_original * height_original * pixelSize,
                           0,
                           &status);
    CHECK_OPENCL_ERROR(status, "clCreateBuffer failed. (inputImageBuffer)");

    // Create memory objects for output Image

	nextImageBuffer = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR | CL_MEM_COPY_HOST_PTR,
		width_original * height_original * pixelSize, inputImageData, &status);

	prevImageBuffer = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR,
		width_original * height_original * pixelSize, 0, &status);



	thetaBuffer = clCreateBuffer(context,
		CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR,
		width * height * pixelSize,0,&status);

    // create a CL program using the kernel source
    buildProgramData buildDataGrey;
	buildDataGrey.kernelName = std::string(GREYSCALE_KERNEL);
	buildDataGrey.devices = devices;
	buildDataGrey.deviceId = sdkContext->deviceId;
	buildDataGrey.flagsStr = std::string("");
    if(sdkContext->isLoadBinaryEnabled())
    {
		buildDataGrey.binaryName = std::string(sdkContext->loadBinary.c_str());
    }

    if(sdkContext->isComplierFlagsSpecified())
    {
		buildDataGrey.flagsFileName = std::string(sdkContext->flags.c_str());
    }

	retValue = buildOpenCLProgram(programGrey, context, buildDataGrey);
    CHECK_ERROR(retValue, 0, "buildOpenCLProgram() failed");

    // get a kernel object handle for a kernel with the given name
    kernelGrey = clCreateKernel(
				  programGrey,
                 "greyscale_filter",
                 &status);
    CHECK_OPENCL_ERROR(status, "clCreateKernel failed.");

	status = kernelInfo.setKernelWorkGroupInfo(kernelGrey,
             devices[sdkContext->deviceId]);
    CHECK_ERROR(status, SDK_SUCCESS,"kernelInfo.setKernelWorkGroupInfo() failed");


    if((blockSizeX * blockSizeY) > kernelInfo.kernelWorkGroupSize)
    {
        if(!sdkContext->quiet)
        {
            std::cout << "Out of Resources!" << std::endl;
            std::cout << "Group Size specified : "
                      << blockSizeX * blockSizeY << std::endl;
            std::cout << "Max Group Size supported on the kernel : "
                      << kernelWorkGroupSize << std::endl;
            std::cout << "Falling back to " << kernelInfo.kernelWorkGroupSize << std::endl;
        }

        // Three possible cases
        if(blockSizeX > kernelInfo.kernelWorkGroupSize)
        {
            blockSizeX = kernelInfo.kernelWorkGroupSize;
            blockSizeY = 1;
        }
    }

	buildProgramData buildDataGaus;
	buildDataGaus.kernelName = std::string(GAUSSIAN_KERNEL);
	buildDataGaus.devices = devices;
	buildDataGaus.deviceId = sdkContext->deviceId;
	buildDataGaus.flagsStr = std::string("");
	if (sdkContext->isLoadBinaryEnabled())
	{
		buildDataGaus.binaryName = std::string(sdkContext->loadBinary.c_str());
	}

	if (sdkContext->isComplierFlagsSpecified())
	{
		buildDataGaus.flagsFileName = std::string(sdkContext->flags.c_str());
	}

	retValue = buildOpenCLProgram(programGaus, context, buildDataGaus);
	CHECK_ERROR(retValue, 0, "buildOpenCLProgram() failed");

	// get a kernel object handle for a kernel with the given name
	kernelGaus = clCreateKernel(
		programGaus,
		"gaussian_filter",
		&status);
	CHECK_OPENCL_ERROR(status, "clCreateKernel failed.");

	status = kernelInfo.setKernelWorkGroupInfo(kernelGaus,
		devices[sdkContext->deviceId]);
	CHECK_ERROR(status, SDK_SUCCESS, "kernelInfo.setKernelWorkGroupInfo() failed");


	if ((blockSizeX * blockSizeY) > kernelInfo.kernelWorkGroupSize)
	{
		if (!sdkContext->quiet)
		{
			std::cout << "Out of Resources!" << std::endl;
			std::cout << "Group Size specified : "
				<< blockSizeX * blockSizeY << std::endl;
			std::cout << "Max Group Size supported on the kernel : "
				<< kernelWorkGroupSize << std::endl;
			std::cout << "Falling back to " << kernelInfo.kernelWorkGroupSize << std::endl;
		}

		// Three possible cases
		if (blockSizeX > kernelInfo.kernelWorkGroupSize)
		{
			blockSizeX = kernelInfo.kernelWorkGroupSize;
			blockSizeY = 1;
		}
	}

	buildProgramData buildDataSobel;
	buildDataSobel.kernelName = std::string(SOBEL_FILTER_KERNEL);
	buildDataSobel.devices = devices;
	buildDataSobel.deviceId = sdkContext->deviceId;
	buildDataSobel.flagsStr = std::string("");
	if (sdkContext->isLoadBinaryEnabled())
	{
		buildDataSobel.binaryName = std::string(sdkContext->loadBinary.c_str());
	}

	if (sdkContext->isComplierFlagsSpecified())
	{
		buildDataSobel.flagsFileName = std::string(sdkContext->flags.c_str());
	}

	retValue = buildOpenCLProgram(programSobel, context, buildDataSobel);
	CHECK_ERROR(retValue, 0, "buildOpenCLProgram() failed");

	// get a kernel object handle for a kernel with the given name
	kernelSobel = clCreateKernel(
		programSobel,
		"sobel_filter",
		&status);
	CHECK_OPENCL_ERROR(status, "clCreateKernel failed.");

	status = kernelInfo.setKernelWorkGroupInfo(kernelSobel,
		devices[sdkContext->deviceId]);
	CHECK_ERROR(status, SDK_SUCCESS, "kernelInfo.setKernelWorkGroupInfo() failed");


	if ((blockSizeX * blockSizeY) > kernelInfo.kernelWorkGroupSize)
	{
		if (!sdkContext->quiet)
		{
			std::cout << "Out of Resources!" << std::endl;
			std::cout << "Group Size specified : "
				<< blockSizeX * blockSizeY << std::endl;
			std::cout << "Max Group Size supported on the kernel : "
				<< kernelWorkGroupSize << std::endl;
			std::cout << "Falling back to " << kernelInfo.kernelWorkGroupSize << std::endl;
		}

		// Three possible cases
		if (blockSizeX > kernelInfo.kernelWorkGroupSize)
		{
			blockSizeX = kernelInfo.kernelWorkGroupSize;
			blockSizeY = 1;
		}
	}

	buildProgramData buildDataMax;
	buildDataMax.kernelName = std::string(MAX_KERNEL);
	buildDataMax.devices = devices;
	buildDataMax.deviceId = sdkContext->deviceId;
	buildDataMax.flagsStr = std::string("");
	if (sdkContext->isLoadBinaryEnabled())
	{
		buildDataMax.binaryName = std::string(sdkContext->loadBinary.c_str());
	}

	if (sdkContext->isComplierFlagsSpecified())
	{
		buildDataMax.flagsFileName = std::string(sdkContext->flags.c_str());
	}

	retValue = buildOpenCLProgram(programMax, context, buildDataMax);
	CHECK_ERROR(retValue, 0, "buildOpenCLProgram() failed");

	// get a kernel object handle for a kernel with the given name
	kernelMax = clCreateKernel(
		programMax,
		"Max_filter",
		&status);
	CHECK_OPENCL_ERROR(status, "clCreateKernel failed.");

	status = kernelInfo.setKernelWorkGroupInfo(kernelMax,
		devices[sdkContext->deviceId]);
	CHECK_ERROR(status, SDK_SUCCESS, "kernelInfo.setKernelWorkGroupInfo() failed");


	if ((blockSizeX * blockSizeY) > kernelInfo.kernelWorkGroupSize)
	{
		if (!sdkContext->quiet)
		{
			std::cout << "Out of Resources!" << std::endl;
			std::cout << "Group Size specified : "
				<< blockSizeX * blockSizeY << std::endl;
			std::cout << "Max Group Size supported on the kernel : "
				<< kernelWorkGroupSize << std::endl;
			std::cout << "Falling back to " << kernelInfo.kernelWorkGroupSize << std::endl;
		}

		// Three possible cases
		if (blockSizeX > kernelInfo.kernelWorkGroupSize)
		{
			blockSizeX = kernelInfo.kernelWorkGroupSize;
			blockSizeY = 1;
		}
	}


	buildProgramData buildDataHyst;
	buildDataHyst.kernelName = std::string(HYSTERESIS_KERNEL);
	buildDataHyst.devices = devices;
	buildDataHyst.deviceId = sdkContext->deviceId;
	buildDataHyst.flagsStr = std::string("");
	if (sdkContext->isLoadBinaryEnabled())
	{
		buildDataHyst.binaryName = std::string(sdkContext->loadBinary.c_str());
	}

	if (sdkContext->isComplierFlagsSpecified())
	{
		buildDataHyst.flagsFileName = std::string(sdkContext->flags.c_str());
	}

	retValue = buildOpenCLProgram(programHyst, context, buildDataHyst);
	CHECK_ERROR(retValue, 0, "buildOpenCLProgram() failed");

	// get a kernel object handle for a kernel with the given name
	kernelHyst = clCreateKernel(
		programHyst,
		"Hyst_filter",
		&status);
	CHECK_OPENCL_ERROR(status, "clCreateKernel failed.");

	status = kernelInfo.setKernelWorkGroupInfo(kernelHyst,
		devices[sdkContext->deviceId]);
	CHECK_ERROR(status, SDK_SUCCESS, "kernelInfo.setKernelWorkGroupInfo() failed");


	if ((blockSizeX * blockSizeY) > kernelInfo.kernelWorkGroupSize)
	{
		if (!sdkContext->quiet)
		{
			std::cout << "Out of Resources!" << std::endl;
			std::cout << "Group Size specified : "
				<< blockSizeX * blockSizeY << std::endl;
			std::cout << "Hyst Group Size supported on the kernel : "
				<< kernelWorkGroupSize << std::endl;
			std::cout << "Falling back to " << kernelInfo.kernelWorkGroupSize << std::endl;
		}

		// Three possible cases
		if (blockSizeX > kernelInfo.kernelWorkGroupSize)
		{
			blockSizeX = kernelInfo.kernelWorkGroupSize;
			blockSizeY = 1;
		}
	}
    return SDK_SUCCESS;
}


int EdgeDetector::GreyScale()
{
	cl_int status;

	// Set input data

	// Set appropriate arguments to the kernel

	// input buffer image
	status = clSetKernelArg(
		kernelGrey,
		0,
		sizeof(cl_mem),
		&nextImageBuffer);
	CHECK_OPENCL_ERROR(status, "clSetKernelArg failed. (nextImageBuffer)")

		// outBuffer imager
		status = clSetKernelArg(
		kernelGrey,
		1,
		sizeof(cl_mem),
		&prevImageBuffer);
	CHECK_OPENCL_ERROR(status, "clSetKernelArg failed. (prevImageBuffer");

	// Enqueue a kernel run call.
	size_t globalThreads[] = { width, height };
	size_t localThreads[] = { blockSizeX, blockSizeY };

	cl_event ndrEvt;
	status = clEnqueueNDRangeKernel(
		commandQueue,
		kernelGrey,
		2,
		NULL,
		globalThreads,
		localThreads,
		0,
		NULL,
		&ndrEvt);
	CHECK_OPENCL_ERROR(status, "clEnqueueNDRangeKernel failed.");

	return 1;
}

int EdgeDetector::Gaussian()
{
	cl_int status;

	// Set input data

	// Set appropriate arguments to the kernel

	// input buffer image
	status = clSetKernelArg(
		kernelGaus,
		0,
		sizeof(cl_mem),
		&prevImageBuffer);
	CHECK_OPENCL_ERROR(status, "clSetKernelArg failed. (PrevBuff())")

		// outBuffer imager
		status = clSetKernelArg(
		kernelGaus,
		1,
		sizeof(cl_mem),
		&nextImageBuffer);
	CHECK_OPENCL_ERROR(status, "clSetKernelArg failed. (outputImageBuffer)");

	// Enqueue a kernel run call.
	size_t globalThreads[] = { width, height };
	size_t localThreads[] = { blockSizeX, blockSizeY };

	cl_event ndrEvt;
	status = clEnqueueNDRangeKernel(
		commandQueue,
		kernelGaus,
		2,
		NULL,
		globalThreads,
		localThreads,
		0,
		NULL,
		&ndrEvt);
	CHECK_OPENCL_ERROR(status, "clEnqueueNDRangeKernel failed.");
	return 1;
}

int EdgeDetector::Sobel()
{
	cl_int status;

	// Set input data

	// Set appropriate arguments to the kernel

	// input buffer image
	status = clSetKernelArg(
		kernelSobel,
		0,
		sizeof(cl_mem),
		&nextImageBuffer);
	CHECK_OPENCL_ERROR(status, "clSetKernelArg failed. (inputImageBuffer)")

		// outBuffer imager
		status = clSetKernelArg(
		kernelSobel,
		1,
		sizeof(cl_mem),
		&prevImageBuffer);
	CHECK_OPENCL_ERROR(status, "clSetKernelArg failed. (outputImageBuffer)");

	status = clSetKernelArg(
		kernelSobel,
		2,
		sizeof(cl_mem),
		&thetaBuffer);
	CHECK_OPENCL_ERROR(status, "clSetKernelArg failed. (thetaBuffer)");

	// Enqueue a kernel run call.
	size_t globalThreads[] = { width, height };
	size_t localThreads[] = { blockSizeX, blockSizeY };
	std::cout << "Width " << width << std::endl;
	std::cout << "height " << height << std::endl;
	cl_event ndrEvt;
	status = clEnqueueNDRangeKernel(
		commandQueue,
		kernelSobel,
		2,
		NULL,
		globalThreads,
		localThreads,
		0,
		NULL,
		&ndrEvt);

	CHECK_OPENCL_ERROR(status, "clEnqueueNDRangeKernel failed.");
	return 1;
}

int EdgeDetector::Max()
{
	cl_int status;

	// Set input data

	// Set appropriate arguments to the kernel

	// input buffer image
	status = clSetKernelArg(
		kernelMax,
		0,
		sizeof(cl_mem),
		&prevImageBuffer);
	CHECK_OPENCL_ERROR(status, "clSetKernelArg failed. (inputImageBuffer)")

		// outBuffer imager
		status = clSetKernelArg(
		kernelMax,
		1,
		sizeof(cl_mem),
		&nextImageBuffer);
	CHECK_OPENCL_ERROR(status, "clSetKernelArg failed. (outputImageBuffer)");

	status = clSetKernelArg(
		kernelMax,
		2,
		sizeof(cl_mem),
		&thetaBuffer);
	CHECK_OPENCL_ERROR(status, "clSetKernelArg failed. (thetaBuffer)");

	// Enqueue a kernel run call.
	size_t globalThreads[] = { width, height };
	size_t localThreads[] = { blockSizeX, blockSizeY };

	cl_event ndrEvt;
	status = clEnqueueNDRangeKernel(
		commandQueue,
		kernelMax,
		2,
		NULL,
		globalThreads,
		localThreads,
		0,
		NULL,
		&ndrEvt);
	CHECK_OPENCL_ERROR(status, "clEnqueueNDRangeKernel failed.");
	return 1;
}
int EdgeDetector::Hysteresis()
{
	cl_int status;


	// Set appropriate arguments to the kernel

	// input buffer image
	status = clSetKernelArg(
		kernelHyst,
		0,
		sizeof(cl_mem),
		&nextImageBuffer);
	CHECK_OPENCL_ERROR(status, "clSetKernelArg failed. (inputImageBuffer)")

		
		// outBuffer imager
		status = clSetKernelArg(
		kernelHyst,
		1,
		sizeof(cl_mem),
		&prevImageBuffer);

	CHECK_OPENCL_ERROR(status, "clSetKernelArg failed. (outputImageBuffer)");

	// Enqueue a kernel run call.
	size_t globalThreads[] = { width, height };
	size_t localThreads[] = { blockSizeX, blockSizeY };

	cl_event ndrEvt;
	status = clEnqueueNDRangeKernel(
		commandQueue,
		kernelHyst,
		2,
		NULL,
		globalThreads,
		localThreads,
		0,
		NULL,
		&ndrEvt);
	CHECK_OPENCL_ERROR(status, "clEnqueueNDRangeKernel failed.");
	return 1;
}
int
EdgeDetector::runCLKernels()
{
	cl_int status;
	EdgeDetector::GreyScale();
	EdgeDetector::Gaussian();
	EdgeDetector::Sobel();
	EdgeDetector::Max();
	EdgeDetector::Hysteresis();

    // Enqueue readBuffer
    cl_event readEvt;
    status = clEnqueueReadBuffer(
                 commandQueue,
                 prevImageBuffer,
                 CL_TRUE,
                 0,
                 width * height * pixelSize,
                 outputImageData,
                 0,
                 NULL,
                 &readEvt);
    CHECK_OPENCL_ERROR(status, "clEnqueueReadBuffer failed.");

    status = clFlush(commandQueue);
    CHECK_OPENCL_ERROR(status, "clFlush failed.");

    status = waitForEventAndRelease(&readEvt);
    CHECK_ERROR(status, SDK_SUCCESS, "WaitForEventAndRelease(readEvt) Failed");

    return SDK_SUCCESS;
}



int
EdgeDetector::initialize()
{
    cl_int status = 0;
    // Call base class Initialize to get default configuration
    status = sdkContext->initialize();
    CHECK_ERROR(status, SDK_SUCCESS, "OpenCL Initialization failed");

    Option* iteration_option = new Option;
    CHECK_ALLOCATION(iteration_option, "Memory Allocation error.\n");

    iteration_option->_sVersion = "i";
    iteration_option->_lVersion = "iterations";
    iteration_option->_description = "Number of iterations to execute kernel";
    iteration_option->_type = CA_ARG_INT;
    iteration_option->_value = &iterations;

    sdkContext->AddOption(iteration_option);

    delete iteration_option;

    return SDK_SUCCESS;
}

int
EdgeDetector::setup()
{
    cl_int status = 0;
    // Allocate host memory and read input image
    std::string filePath = getPath() + std::string(INPUT_IMAGE);
	std::cout << "Input File:  " << filePath << std::endl;
    status = readInputImage(filePath);
    CHECK_ERROR(status, SDK_SUCCESS, "Read InputImage failed");

    // create and initialize timers
    int timer = sampleTimer->createTimer();
    sampleTimer->resetTimer(timer);
    sampleTimer->startTimer(timer);

    status = setupCL();
    if(status != SDK_SUCCESS)
    {
        return status;
    }

    sampleTimer->stopTimer(timer);
    // Compute setup time
    setupTime = (double)(sampleTimer->readTimer(timer));

    return SDK_SUCCESS;
}


int
EdgeDetector::run()
{
    cl_int status = 0;
    if(!byteRWSupport)
    {
        return SDK_SUCCESS;
    }

    for(int i = 0; i < 2 && iterations != 1; i++)
    {
        // Set kernel arguments and run kernel
        if(runCLKernels() != SDK_SUCCESS)
        {
            return SDK_FAILURE;
        }
    }

    std::cout << "Executing kernel for " << iterations
              << " iterations" <<std::endl;
    std::cout << "-------------------------------------------" << std::endl;

    // create and initialize timers
    int timer = sampleTimer->createTimer();
    sampleTimer->resetTimer(timer);
    sampleTimer->startTimer(timer);

    for(int i = 0; i < iterations; i++)
    {
        // Set kernel arguments and run kernel
        if(runCLKernels() != SDK_SUCCESS)
        {
            return SDK_FAILURE;
        }
    }

    sampleTimer->stopTimer(timer);
    // Compute kernel time
    kernelTime = (double)(sampleTimer->readTimer(timer)) / iterations;

    // write the output image to bitmap file
    status = writeOutputImage(OUTPUT_IMAGE);
    CHECK_ERROR(status, SDK_SUCCESS, "write Output Image Failed");

    return SDK_SUCCESS;
}

int
EdgeDetector::cleanup()
{
    if(!byteRWSupport)
    {
        return SDK_SUCCESS;
    }

    // Releases OpenCL resources (Context, Memory etc.)
    cl_int status;

    status = clReleaseKernel(kernelGrey);
    CHECK_OPENCL_ERROR(status, "clReleaseKernel failed.");

    status = clReleaseProgram(programGrey);
    CHECK_OPENCL_ERROR(status, "clReleaseProgram failed.");

	status = clReleaseProgram(programGaus);
	CHECK_OPENCL_ERROR(status, "clReleaseProgram failed.");

	status = clReleaseKernel(kernelGaus);
	CHECK_OPENCL_ERROR(status, "clReleaseKernel failed.");

	status = clReleaseProgram(programSobel);
	CHECK_OPENCL_ERROR(status, "clReleaseProgram failed.");

	status = clReleaseKernel(kernelSobel);
	CHECK_OPENCL_ERROR(status, "clReleaseKernel failed.");

	status = clReleaseProgram(programMax);
	CHECK_OPENCL_ERROR(status, "clReleaseProgram failed.");

	status = clReleaseKernel(kernelMax);
	CHECK_OPENCL_ERROR(status, "clReleaseKernel failed.");

	status = clReleaseProgram(programHyst);
	CHECK_OPENCL_ERROR(status, "clReleaseProgram failed.");

	status = clReleaseKernel(kernelHyst);
	CHECK_OPENCL_ERROR(status, "clReleaseKernel failed.");

    status = clReleaseMemObject(inputImageBuffer);
    CHECK_OPENCL_ERROR(status, "clReleaseMemObject failed.");

    status = clReleaseMemObject(nextImageBuffer);
    CHECK_OPENCL_ERROR(status, "clReleaseMemObject failed.");

	status = clReleaseMemObject(prevImageBuffer);
	CHECK_OPENCL_ERROR(status, "clReleaseMemObject failed.");

    status = clReleaseCommandQueue(commandQueue);
    CHECK_OPENCL_ERROR(status, "clReleaseCommandQueue failed.");

    status = clReleaseContext(context);
    CHECK_OPENCL_ERROR(status, "clReleaseContext failed.");

    // release program resources (input memory etc.)
    FREE(inputImageData);

    FREE(outputImageData);

	//FREE(nextImageData);

    FREE(verificationOutput);

    FREE(devices);

    return SDK_SUCCESS;
}


void
EdgeDetector::EdgeDetectorCPUReference()
{
    // x-axis gradient mask
    const int kx[][3] =
    {
        { 1, 2, 1},
        { 0, 0, 0},
        { -1,-2,-1}
    };

    // y-axis gradient mask
    const int ky[][3] =
    {
        { 1, 0, -1},
        { 2, 0, -2},
        { 1, 0, -1}
    };

    int gx = 0;
    int gy = 0;

    // pointer to input image data
    cl_uchar *ptr = (cl_uchar*)malloc(width * height * pixelSize);
    memcpy(ptr, inputImageData, width * height * pixelSize);

    // each pixel has 4 uchar components
    int w = width * 4;

    int k = 1;

    // apply filter on each pixel (except boundary pixels)
    for(int i = 0; i < (int)(w * (height - 1)) ; i++)
    {
        if(i < (k+1)*w - 4 && i >= 4 + k*w)
        {
            gx =  kx[0][0] **(ptr + i - 4 - w)
                  + kx[0][1] **(ptr + i - w)
                  + kx[0][2] **(ptr + i + 4 - w)
                  + kx[1][0] **(ptr + i - 4)
                  + kx[1][1] **(ptr + i)
                  + kx[1][2] **(ptr + i + 4)
                  + kx[2][0] **(ptr + i - 4 + w)
                  + kx[2][1] **(ptr + i + w)
                  + kx[2][2] **(ptr + i + 4 + w);


            gy =  ky[0][0] **(ptr + i - 4 - w)
                  + ky[0][1] **(ptr + i - w)
                  + ky[0][2] **(ptr + i + 4 - w)
                  + ky[1][0] **(ptr + i - 4)
                  + ky[1][1] **(ptr + i)
                  + ky[1][2] **(ptr + i + 4)
                  + ky[2][0] **(ptr + i - 4 + w)
                  + ky[2][1] **(ptr + i + w)
                  + ky[2][2] **(ptr + i + 4 + w);

            float gx2 = pow((float)gx, 2);
            float gy2 = pow((float)gy, 2);


            *(verificationOutput + i) = (cl_uchar)(sqrt(gx2 + gy2) / 2.0);
        }

        // if reached at the end of its row then incr k
        if(i == (k + 1) * w - 5)
        {
            k++;
        }
    }

    free(ptr);
}

/*
int
EdgeDetector::verifyResults()
{
    if(!byteRWSupport)
    {
        return SDK_SUCCESS;
    }

    if(sdkContext->verify)
    {
        // reference implementation
        EdgeDetectorCPUReference();

        float *outputDevice = new float[width * height * pixelSize];
        CHECK_ALLOCATION(outputDevice,
                         "Failed to allocate host memory! (outputDevice)");

        float *outputReference = new float[width * height * pixelSize];
        CHECK_ALLOCATION(outputReference, "Failed to allocate host memory!"
                         "(outputReference)");

        // copy uchar data to float array
        for(int i = 0; i < (int)(width * height); i++)
        {
            outputDevice[i * 4 + 0] = outputImageData[i].s[0];
            outputDevice[i * 4 + 1] = outputImageData[i].s[1];
            outputDevice[i * 4 + 2] = outputImageData[i].s[2];
            outputDevice[i * 4 + 3] = outputImageData[i].s[3];

            outputReference[i * 4 + 0] = verificationOutput[i * 4 + 0];
            outputReference[i * 4 + 1] = verificationOutput[i * 4 + 1];
            outputReference[i * 4 + 2] = verificationOutput[i * 4 + 2];
            outputReference[i * 4 + 3] = verificationOutput[i * 4 + 3];
        }


        // compare the results and see if they match
        if(compare(outputReference,
                   outputDevice,
                   width * height * 4))
        {
            std::cout << "Passed!\n" << std::endl;
            delete[] outputDevice;
            delete[] outputReference;
            return SDK_SUCCESS;
        }
        else
        {
            std::cout << "Failed\n" << std::endl;
            delete[] outputDevice;
            delete[] outputReference;
            return SDK_FAILURE;
        }
    }

    return SDK_SUCCESS;
}
*/
void
EdgeDetector::printStats()
{
    if(sdkContext->timing)
    {
        std::string strArray[4] =
        {
            "Width",
            "Height",
            "Time(sec)",
            "[Transfer+Kernel]Time(sec)"
        };
        std::string stats[4];

        sampleTimer->totalTime = setupTime + kernelTime;

        stats[0] = toString(width, std::dec);
        stats[1] = toString(height, std::dec);
        stats[2] = toString(sampleTimer->totalTime, std::dec);
        stats[3] = toString(kernelTime, std::dec);

        printStatistics(strArray, stats, 4);
    }
}

