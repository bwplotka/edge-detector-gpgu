__kernel void Max_filter(__global uchar4* inputImage, __global uchar4* outputImage, __global uchar* theta)
{
	uint x = get_global_id(0);
	uint y = get_global_id(1);

	uint width = get_global_size(0);
	uint height = get_global_size(1);


	int c = x + y * width;

	uchar4 magnitude = inputImage[c];

	switch (theta[c])
	{
	case 0:
		if (magnitude.x <= inputImage[c - 1].x || magnitude.x <= inputImage[c + 1].x)
		{
			outputImage[c] = 0;//convert_uchar4(0);
		}
		else
		{
			outputImage[c] = magnitude;//convert_uchar4(magnitude);
		}
		break;

	case 45:
		if (magnitude.x <= inputImage[c + 1 - width].x || magnitude.x <= inputImage[c - 1 + width].x)
		{
			outputImage[c] = 0;//convert_uchar4(0);
		}
		else
		{
			outputImage[c] = magnitude;//convert_uchar4(magnitude);
		}
		break;

	case 90:
		if (magnitude.x <= inputImage[c - width].x || magnitude.x <= inputImage[c + width].x)
		{
			outputImage[c] = 0;//convert_uchar4(0);
		}
		else
		{
			outputImage[c] = magnitude;//convert_uchar4(magnitude);
		}
		break;

	case 135:
		if (magnitude.x <= inputImage[c - 1 - width].x || magnitude.x <= inputImage[c + 1 + width].x)
		{
			outputImage[c] = 0;//convert_uchar4(0);
		}
		else
		{
			outputImage[c] = magnitude;//convert_uchar4(magnitude);
		}
		break;
	}



}
