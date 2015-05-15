#include "EdgeDetector.hpp"

int
main(int argc, char * argv[])
{
	cl_int status = 0;
	EdgeDetector clEdgeDetector;

	if (clEdgeDetector.initialize() != SDK_SUCCESS)
	{
		return SDK_FAILURE;
	}

	if (clEdgeDetector.sdkContext->parseCommandLine(argc, argv) != SDK_SUCCESS)
	{
		return SDK_FAILURE;
	}

	if (clEdgeDetector.sdkContext->isDumpBinaryEnabled())
	{
		return clEdgeDetector.genBinaryImage();
	}

	std::string filePath = getPath() + std::string(INPUT_IMAGE);
	std::cout << "Input File:  " << filePath << std::endl;
	status = clEdgeDetector.readInputImage(filePath);
	CHECK_ERROR(status, SDK_SUCCESS, "Read InputImage failed");
	
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
	status = clEdgeDetector.writeOutputImage(OUTPUT_IMAGE);
	CHECK_ERROR(status, SDK_SUCCESS, "write Output Image Failed");

	/*if (clEdgeDetector.verifyResults() != SDK_SUCCESS)
	{
		return SDK_FAILURE;
	}*/

	if (clEdgeDetector.cleanup() != SDK_SUCCESS)
	{
		return SDK_FAILURE;
	}

	clEdgeDetector.printStats();
	return SDK_SUCCESS;
}
