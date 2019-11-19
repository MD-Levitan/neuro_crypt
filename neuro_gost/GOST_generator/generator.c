#include <string.h>
#include <stdlib.h>
#include <time.h>

#include "magma.h"

static uint8_t key[32] = { 0x00 };
// 	0x88, 0x99, 0xaa, 0xbb, 0xcc, 0xdd, 0xee, 0xff,
// 	0x00, 0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77,
// 	0xfe, 0xdc, 0xba, 0x98, 0x76, 0x54, 0x32, 0x10,
// 	0x01, 0x23, 0x45, 0x67, 0x89, 0xab, 0xcd, 0xef
// };

const char *usage = "Usage:  ./generator <SIZE> <GEN_TYPE>";

void iterate_generator(struct crypto_tfm *ctx, uint64_t size, 
					   char *filename_x, char *filename_y)
{
	FILE *out_file_x, *out_file_y;
	out_file_x = fopen(filename_x, "wb");
	out_file_y = fopen(filename_y, "wb");	

	if (out_file_x == NULL || out_file_y == NULL) 
	{
		printf("error: cannot open file!\n");
		exit(1);
	}

	uint64_t in, out;
	for (in = 0; in < size; ++in)
	{
		magma_it(ctx, (uint8_t *) &out, (uint8_t *) &in, 0);
		
		fwrite((uint8_t *) &in, sizeof(uint8_t), 8, out_file_x);
		fwrite((uint8_t *) &out, sizeof(uint8_t), 8, out_file_y);
	}

	fclose(out_file_x);
	fclose(out_file_y);	
}


void random_generator(struct crypto_tfm *ctx, uint64_t size, 
					   char *filename_x, char *filename_y)
{
	FILE *out_file_x, *out_file_y;
	out_file_x = fopen(filename_x, "wb");
	out_file_y = fopen(filename_y, "wb");	

	uint64_t in, out;
	srand(time(NULL));
	
	for (uint64_t i = 0; i < size; ++i)
	{
		in = 0;
		((uint32_t *) &in)[0] = rand();
		((uint32_t *) &in)[1] = rand();
		magma_it(ctx, (uint8_t *) &out, (uint8_t *) &in, 0);
		fwrite((uint8_t *) &in, sizeof(uint8_t), 8, out_file_x);
		fwrite((uint8_t *) &out, sizeof(uint8_t), 8, out_file_y);
	}

	fclose(out_file_x);
	fclose(out_file_y);	
}

void random_generator_n(struct crypto_tfm *ctx, uint64_t size, 
					   uint8_t n, char *filename_x, char *filename_y)
{
	FILE *out_file_x, *out_file_y;
	out_file_x = fopen(filename_x, "wb");
	out_file_y = fopen(filename_y, "wb");	

	uint64_t in, out;
	srand(time(NULL));
	
	for (uint64_t i = 0; i < size; ++i)
	{
		in = 0;
		((uint32_t *) &in)[0] = rand();
		((uint32_t *) &in)[1] = rand();
		magma_it_n(ctx, (uint8_t *) &out, (uint8_t *) &in, n);
		fwrite((uint8_t *) &in, sizeof(uint8_t), 8, out_file_x);
		fwrite((uint8_t *) &out, sizeof(uint8_t), 8, out_file_y);
	}
	
	fclose(out_file_x);
	fclose(out_file_y);	
}

void random_generator_primitive(struct crypto_tfm *ctx, uint64_t size,
								char *filename_x, char *filename_y)
{
	FILE *out_file_x, *out_file_y;
	out_file_x = fopen(filename_x, "wb");
	out_file_y = fopen(filename_y, "wb");	

	uint64_t in, out;
	uint32_t x, y;
	srand(time(NULL));
	
	for (uint64_t i = 0; i < size; ++i)
	{
		in = 0;
		((uint32_t *) &in)[0] = rand();
		((uint32_t *) &in)[1] = rand();
		magma_neuro(ctx, (uint8_t *) &out, (uint8_t *) &in, &x, &y);

		// printf("%llx\n", x);
		// printf("%llx\n", y);
		// printf("%llx\n", in);
		// printf("%llx\n", out);

		uint8_t p = 0; 
		//for (uint8_t p = 0; p < 8; ++p)
		//{
			uint8_t var = ((uint8_t *) &in)[4 + p / 2];
			var = p % 2 ?  (var & 0xF) : (var >> 4 & 0xF);

			uint8_t var1 = ((uint8_t *) &x)[p / 2];
			var1 = p % 2 ?  (var1 & 0xF) : (var1 >> 4 & 0xF);
			var = (var << 4) | var1;

			uint8_t var2 = ((uint8_t *) &y)[p / 2];
			var2 = p % 2 ?  (var2 & 0xF) : (var2 >> 4 & 0xF);
			
			fwrite(&var, sizeof(uint8_t), 1, out_file_x);
			fwrite(&var2, sizeof(uint8_t), 1, out_file_y);
		//}
	}

	fclose(out_file_x);
	fclose(out_file_y);	
}



/*
 *	Generator
 *  Usage:  ./generator <SIZE> <GEN_TYPE>
 *
 */
int main(int argc, const char **argv) {
	if (argc < 3) 
	{
		printf("%s!\n", usage);
		return -1;
	}
	
	int8_t rv;
	char *filename_y = "out_y.bin", *filename_x = "out_x.bin";
	uint64_t size = 0;
	FILE *out_file_x, *out_file_y;
	
	struct crypto_tfm *ctx;
	
	size = strtoll(argv[1], NULL, 10);
	rv = strtol(argv[2], NULL, 10);

	
	ctx = create_tfm_ctx();
	magma_setkey(ctx, key, sizeof(key));
	switch(rv)
	{ 
		case 2: 
		{
			random_generator(ctx, size, 
							"bin/out_random_x.bin",
							"bin/out_random_y.bin");
			break;
		}
		case 1: 
		{ 
			iterate_generator(ctx, size,
							 "bin/out_iterate_x.bin",
							 "bin/out_iterate_y.bin");
			break;
		}
		case 3:
		{
			random_generator_n(ctx, size, 2,
							  "bin/out_random_n_x.bin",
							  "bin/out_random_n_y.bin");
			break;
		}
		case 4:
		{
			random_generator_primitive(ctx, size,
									  "bin/out_primitive_x.bin",
									  "bin/out_primitive_y.bin");
		}
	}
	

	delete_tfm_ctx(ctx);
}

// // test
// int main()
// {
// 	struct crypto_tfm *ctx;
// 	ctx = create_tfm_ctx();
// 	magma_setkey(ctx, key, sizeof(key));

// 	uint64_t in, out1, out2, y, x;
// 	((uint32_t *) &in)[0] = 11111111;
// 	((uint32_t *) &in)[1] = 22222222;

// 	magma_neuro(ctx, &out2, &in, &y, &x);
// 	_magma_it(ctx, &out1, &in, 0);

// 	printf("First: %llx\n", out1);
// 	printf("Second: %llx\n", out2);
// }