#include <string.h>
#include <stdlib.h>
#include <time.h>

#include "magma.h"

static uint8_t key[32] = {
	0x88, 0x99, 0xaa, 0xbb, 0xcc, 0xdd, 0xee, 0xff,
	0x00, 0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77,
	0xfe, 0xdc, 0xba, 0x98, 0x76, 0x54, 0x32, 0x10,
	0x01, 0x23, 0x45, 0x67, 0x89, 0xab, 0xcd, 0xef
};

const char *usage = "Usage:  ./generator <SIZE>";

void iterate_generator(struct crypto_tfm *ctx, uint64_t size, 
					   FILE *out_file_x, FILE *out_file_y)
{
	uint64_t in, out;
	for (in = 0; in < size; ++in)
	{
		//out = in * 2;
		magma_it(ctx, (uint8_t *) &out, (uint8_t *) &in, 0);
		fwrite((uint8_t *) &in, sizeof(uint8_t), 8, out_file_x);
		fwrite((uint8_t *) &out, sizeof(uint8_t), 8, out_file_y);
	}
}


void random_generator(struct crypto_tfm *ctx, uint64_t size, 
					   FILE *out_file_x, FILE *out_file_y)
{
	uint64_t in, out;
	srand(time(NULL));
	
	for (uint64_t i = 0; i < size; ++i)
	{
		in = rand();
		//out = in * 2;
		magma_it(ctx, (uint8_t *) &out, (uint8_t *) &in, 0);
		fwrite((uint8_t *) &in, sizeof(uint8_t), 8, out_file_x);
		fwrite((uint8_t *) &out, sizeof(uint8_t), 8, out_file_y);
	}
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
	
	out_file_x = fopen(filename_x, "wb");
	out_file_y = fopen(filename_y, "wb");	
	
	if (out_file_x == NULL || out_file_y == NULL) 
	{
		printf("error: cannot open file!\n");
		return -1;
	}
	
	ctx = create_tfm_ctx();
	magma_setkey(ctx, key, sizeof(key));
	if (rv == 2) {
		random_generator(ctx, size, out_file_x, out_file_y);
	} else {
		iterate_generator(ctx, size, out_file_x, out_file_y);
	}
	
	fclose(out_file_x);
	fclose(out_file_y);	
	delete_tfm_ctx(ctx);
}
