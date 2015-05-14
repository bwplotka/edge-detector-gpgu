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

	status = clEdgeDetector.setup();
	if (status != SDK_SUCCESS)
	{
		return status;
	}

	if (clEdgeDetector.run() != SDK_SUCCESS)
	{
		return SDK_FAILURE;
	}

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
