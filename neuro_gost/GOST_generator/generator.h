#ifndef GENERATOR_H_
#define GENERATOR_H_

#include <string.h>
#include <stdlib.h>
#include <time.h>
#include <inttypes.h>
#include <stdio.h>

#include "magma.h"
#include "feistel.h"

typedef union crypto_tfm crypto_tfm;

union crypto_tfm {
    crypto_magma_ctx *magma;
    crypto_feistel_ctx *feistel;
};

typedef void (*generator)(crypto_tfm *ctx, uint64_t size,
                          char *filename_x, char *filename_y,
                          void (*generator)(crypto_tfm *ctx, FILE *out_file_x, FILE *out_file_y, uint64_t in));

typedef void (*generator_model)(crypto_tfm *ctx, FILE *out_file_x, FILE *out_file_y, uint64_t in);

/**
 * @brief Iterate from 0 to @size as input to @gen function
 * 
 * @param ctx           Magma context
 * @param size          size of output sequence <= UINT64_MAX
 * @param filename_x    filename with input
 * @param filename_y    filename with output
 * @param gen           generator of output sequence
 */
void iterate_generator(crypto_tfm *ctx, uint64_t size,
                       char *filename_x, char *filename_y,
                       generator_model gen);

/**
 * @brief Iterate from 0 to @size in two 32-bite blocks as input to @gen function
 * 
 * @param ctx           Magma context
 * @param size          size of output sequence <= UINT32_MAX
 * @param filename_x    filename with input
 * @param filename_y    filename with output
 * @param gen           generator of output sequence
 */
void iterate_parallel_generator(crypto_tfm *ctx, uint64_t size,
                                char *filename_x, char *filename_y,
                                generator_model gen);

/**
 * @brief Generate random @size values as input to @gen function
 * 
 * @param ctx           Magma context
 * @param size          size of output sequence <= UINT64_MAX
 * @param filename_x    filename with input
 * @param filename_y    filename with output
 * @param gen           generator of output sequence
 */
void random_generator(crypto_tfm *ctx, uint64_t size,
                      char *filename_x, char *filename_y,
                      generator_model gen);

/**
 * @brief Iterate values from 0 to @size in random order as input to @gen function
 * 
 * @param ctx           Magma context
 * @param size          size of output sequence <= UINT64_MAX
 * @param filename_x    filename with input
 * @param filename_y    filename with output
 * @param gen           generator of output sequence
 */
void random_iterate_generator(crypto_tfm *ctx, uint64_t size,
                              char *filename_x, char *filename_y,
                              generator_model gen);

/**
 * @brief Generator for 1-round of GOST
 * 
 * @param ctx           Magma context
 * @param out_file_x    File context to write input
 * @param out_file_y    File context to write output
 * @param in            Input value
 */
void round_generator(crypto_tfm *ctx, FILE *out_file_x, FILE *out_file_y, uint64_t in);

/**
 * @brief Generator for 2-round of GOST
 * 
 * @param ctx           Magma context
 * @param out_file_x    File context to write input
 * @param out_file_y    File context to write output
 * @param in            Input value
 */
void n_round_generator(crypto_tfm *ctx, FILE *out_file_x, FILE *out_file_y, uint64_t in);

/**
 * @brief Generator for G0 model. Input - 8 bits, output - 4 bits.
 * 
 * @param ctx           Magma context
 * @param out_file_x    File context to write input
 * @param out_file_y    File context to write output
 * @param in            Input value
 */
void primitive_g0_generator(crypto_tfm *ctx, FILE *out_file_x, FILE *out_file_y, uint64_t in);

/**
 * @brief Generator for G1 model. Input - 8 bits, output - 4 bits.
 * 
 * @param ctx           Magma context
 * @param out_file_x    File context to write input
 * @param out_file_y    File context to write output
 * @param in            Input value
 */
void primitive_g1_generator(crypto_tfm *ctx, FILE *out_file_x, FILE *out_file_y, uint64_t in);

/**
 * @brief Generator for G2 model. Input - 8 bits, output - 4 bits.
 * 
 * @param ctx           Magma context
 * @param out_file_x    File context to write input
 * @param out_file_y    File context to write output
 * @param in            Input value
 */
void primitive_g2_generator(crypto_tfm *ctx, FILE *out_file_x, FILE *out_file_y, uint64_t in);

/**
 * @brief Generator for G3 model. Input - 64 bits, output - 32 bits.
 * 
 * @param ctx           Magma context
 * @param out_file_x    File context to write input
 * @param out_file_y    File context to write output
 * @param in            Input value
 */
void primitive_g3_generator(crypto_tfm *ctx, FILE *out_file_x, FILE *out_file_y, uint64_t in);

/**
 * @brief Generator for Feistel model. Input - 16 bits, output - 16 bits.
 * 
 * @param ctx           Magma context
 * @param out_file_x    File context to write input
 * @param out_file_y    File context to write output
 * @param in            Input value
 */
void feistel_generator(crypto_tfm *ctx, FILE *out_file_x, FILE *out_file_y, uint64_t in);

void generate_random_key(uint8_t *key);

typedef struct generator_type_t generator_type_t;
typedef enum ciphersuite_t ciphersuite_t;

enum ciphersuite_t
{
    MAGMA,
    FEISTEL
};

struct generator_type_t
{
    const char *name;
    union func {
        generator_model gen_model_func;
        generator gen_func;
    } func;
    const char *description;
    const char *default_input;
    const char *default_output;
    ciphersuite_t suite;
};

static generator_type_t generators[] = {
    {.name = "iter", .func.gen_func = iterate_generator, .description = "Iterate from 0 to size"},
    {.name = "iter2", .func.gen_func = iterate_parallel_generator, .description = "Iterate from 0 to size in 2 32-bite blocks"},
    {.name = "rand", .func.gen_func = random_generator, .description = "Generate random value as input"},
    {.name = "rand2", .func.gen_func = random_iterate_generator, .description = "Iterate from 0 to size in random order"},
};

static generator_type_t model_generators[] = {
    {
        .name = "GOST-1",
        .func.gen_model_func = round_generator,
        .description = "Use 1-round GOST encryption",
        .default_input = "bin/n1_x.bin",
        .default_output = "bin/n1_y.bin",
        .suite = MAGMA,
    },
    {
        .name = "GOST-2",
        .func.gen_model_func = n_round_generator,
        .description = "Use 2-round GOST encryption",
        .default_input = "bin/n2_x.bin",
        .default_output = "bin/n2_y.bin",
        .suite = MAGMA,
    },
    {
        .name = "G0",
        .func.gen_model_func = primitive_g0_generator,
        .description = "Use G0 model",
        .default_input = "bin/g0_x.bin",
        .default_output = "bin/g0_y.bin",
        .suite = MAGMA,
    },
    {
        .name = "G1",
        .func.gen_model_func = primitive_g1_generator,
        .description = "Use G1 model",
        .default_input = "bin/g1_x.bin",
        .default_output = "bin/g1_y.bin",
        .suite = MAGMA,
    },
    {
        .name = "G2",
        .func.gen_model_func = primitive_g2_generator,
        .description = "Use G2 model",
        .default_input = "bin/g2_x.bin",
        .default_output = "bin/g2_y.bin",
        .suite = MAGMA,
    },
    {
        .name = "G3",
        .func.gen_model_func = primitive_g3_generator,
        .description = "Use G3 model",
        .default_input = "bin/g3_x.bin",
        .default_output = "bin/g3_y.bin",
        .suite = MAGMA,
    },
    {
        .name = "Feistel",
        .func.gen_model_func = feistel_generator,
        .description = "Use feistel model",
        .default_input = "bin/f_x.bin",
        .default_output = "bin/f_y.bin",
        .suite = FEISTEL,
    },
};

void print_models();
void print_generators();

generator_type_t *get_type_by_name(const char *name, generator_type_t *list, uint8_t size);

#define get_generator_by_name(name) get_type_by_name(name, generators, sizeof(generators) / sizeof(generator_type_t));
#define get_model_by_name(name) get_type_by_name(name, model_generators, sizeof(model_generators) / sizeof(generator_type_t));

#endif //GENERATOR_H_