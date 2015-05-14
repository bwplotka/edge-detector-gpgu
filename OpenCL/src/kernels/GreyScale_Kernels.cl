
__kernel void greyscale_filter(__global uchar4* inputImage, __global uchar4* outputImage)
{
	uint x = get_global_id(0);
	uint y = get_global_id(1);

	uint width = get_global_size(0);
	uint height = get_global_size(1);


	int c = x + y * width;
	uchar4 color = inputImage[c];
	uchar lum = (uchar)(0.30 *color.x + 0.59 *color.y + 0.11 *color.z);
	outputImage[c] = lum;//convert_uchar4(lum);

}