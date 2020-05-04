#ifndef GENERATOR_H_
#define GENERATOR_H_

#include <string.h>
#include <stdlib.h>
#include <time.h>
#include <inttypes.h>
#include <stdio.h>

#include "crypto.h"

typedef void (*generator)(crypto_tfm *ctx, uint64_t size,
                          const char *filename_x, const char *filename_y,
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
                       const char *filename_x, const char *filename_y,
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
                                const char *filename_x, const char *filename_y,
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
                      const char *filename_x, const char *filename_y,
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
                              const char *filename_x, const char *filename_y,
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
typedef struct model_type_t model_type_t;

struct generator_type_t
{
    const char *name;
    generator gen_func;
    const char *description;
};

struct model_type_t
{
    const char *name;
    generator_model gen_model_func;
    const char *description;
    const char *default_input;
    const char *default_output;
    ciphersuite_t suite;
    crypto_params params;
};

static generator_type_t generators[] = {
    {.name = "iter", .gen_func = iterate_generator, .description = "Iterate from 0 to size"},
    {.name = "iter2", .gen_func = iterate_parallel_generator, .description = "Iterate from 0 to size in 2 32-bite blocks"},
    {.name = "rand", .gen_func = random_generator, .description = "Generate random value as input"},
    {.name = "rand2", .gen_func = random_iterate_generator, .description = "Iterate from 0 to size in random order"},
};

static model_type_t models[] = {
    {
        .name = "GOST-1",
        .gen_model_func = round_generator,
        .description = "Use 1-round GOST encryption",
        .default_input = "bin/n1_x.bin",
        .default_output = "bin/n1_y.bin",
        .suite = MAGMA,
    },
    {
        .name = "GOST-2",
        .gen_model_func = n_round_generator,
        .description = "Use 2-round GOST encryption",
        .default_input = "bin/n2_x.bin",
        .default_output = "bin/n2_y.bin",
        .suite = MAGMA,
    },
    {
        .name = "G0",
        .gen_model_func = primitive_g0_generator,
        .description = "Use G0 model",
        .default_input = "bin/g0_x.bin",
        .default_output = "bin/g0_y.bin",
        .suite = MAGMA,
    },
    {
        .name = "G1",
        .gen_model_func = primitive_g1_generator,
        .description = "Use G1 model",
        .default_input = "bin/g1_x.bin",
        .default_output = "bin/g1_y.bin",
        .suite = MAGMA,
    },
    {
        .name = "G2",
        .gen_model_func = primitive_g2_generator,
        .description = "Use G2 model",
        .default_input = "bin/g2_x.bin",
        .default_output = "bin/g2_y.bin",
        .suite = MAGMA,
    },
    {
        .name = "G3",
        .gen_model_func = primitive_g3_generator,
        .description = "Use G3 model",
        .default_input = "bin/g3_x.bin",
        .default_output = "bin/g3_y.bin",
        .suite = MAGMA,
    },
    {
        .name = "F0",
        .gen_model_func = feistel_generator,
        .description = "Use feistel model with 0 shift",
        .default_input = "bin/f0_x.bin",
        .default_output = "bin/f0_y.bin",
        .suite = FEISTEL,
        .params.feistel_params = {.iter = 1, .shift = 1},
    },
    {
        .name = "F1",
        .gen_model_func = feistel_generator,
        .description = "Use feistel model with 1 shift",
        .default_input = "bin/f1_x.bin",
        .default_output = "bin/f1_y.bin",
        .suite = FEISTEL,
        .params.feistel_params = {.iter = 1, .shift = 1},
    },
    {
        .name = "F2",
        .gen_model_func = feistel_generator,
        .description = "Use feistel model with 2 shift",
        .default_input = "bin/f2_x.bin",
        .default_output = "bin/f2_y.bin",
        .suite = FEISTEL,
        .params.feistel_params = {.iter = 1, .shift = 2},
    },
    {
        .name = "F3",
        .gen_model_func = feistel_generator,
        .description = "Use feistel model with 3 shift",
        .default_input = "bin/f3_x.bin",
        .default_output = "bin/f3_y.bin",
        .suite = FEISTEL,
        .params.feistel_params = {.iter = 1, .shift = 3},
    },
    {
        .name = "F4",
        .gen_model_func = feistel_generator,
        .description = "Use feistel model with 0 shift",
        .default_input = "bin/f4_x.bin",
        .default_output = "bin/f4_y.bin",
        .suite = FEISTEL,
        .params.feistel_params = {.iter = 1, .shift = 4},
    },
    {
        .name = "F5",
        .gen_model_func = feistel_generator,
        .description = "Use feistel model with 5 shift",
        .default_input = "bin/f5_x.bin",
        .default_output = "bin/f5_y.bin",
        .suite = FEISTEL,
        .params.feistel_params = {.iter = 1, .shift = 5},
    },
    {
        .name = "F6",
        .gen_model_func = feistel_generator,
        .description = "Use feistel model with 6 shift",
        .default_input = "bin/f6_x.bin",
        .default_output = "bin/f6_y.bin",
        .suite = FEISTEL,
        .params.feistel_params = {.iter = 1, .shift = 6},
    },
    {
        .name = "F7",
        .gen_model_func = feistel_generator,
        .description = "Use feistel model with 7 shift",
        .default_input = "bin/f7_x.bin",
        .default_output = "bin/f7_y.bin",
        .suite = FEISTEL,
        .params.feistel_params = {.iter = 1, .shift = 7},
    },
    {
        .name = "F0-2",
        .gen_model_func = feistel_generator,
        .description = "Use feistel model with 0 shift",
        .default_input = "bin/f0-2_x.bin",
        .default_output = "bin/f0-2_y.bin",
        .suite = FEISTEL,
        .params.feistel_params = {.iter = 2, .shift = 1},
    },
    {
        .name = "F1-2",
        .gen_model_func = feistel_generator,
        .description = "Use feistel model with 1 shift",
        .default_input = "bin/f1-2_x.bin",
        .default_output = "bin/f1-2_y.bin",
        .suite = FEISTEL,
        .params.feistel_params = {.iter = 2, .shift = 1},
    },
    {
        .name = "F2-2",
        .gen_model_func = feistel_generator,
        .description = "Use feistel model with 2 shift",
        .default_input = "bin/f2-2_x.bin",
        .default_output = "bin/f2-2_y.bin",
        .suite = FEISTEL,
        .params.feistel_params = {.iter = 2, .shift = 2},
    },
    {
        .name = "F3-2",
        .gen_model_func = feistel_generator,
        .description = "Use feistel model with 3 shift",
        .default_input = "bin/f3-2_x.bin",
        .default_output = "bin/f3-2_y.bin",
        .suite = FEISTEL,
        .params.feistel_params = {.iter = 2, .shift = 3},
    },
    {
        .name = "F4-2",
        .gen_model_func = feistel_generator,
        .description = "Use feistel model with 0 shift",
        .default_input = "bin/f4-2_x.bin",
        .default_output = "bin/f4-2_y.bin",
        .suite = FEISTEL,
        .params.feistel_params = {.iter = 2, .shift = 4},
    },
    {
        .name = "F5-2",
        .gen_model_func = feistel_generator,
        .description = "Use feistel model with 5 shift",
        .default_input = "bin/f5-2_x.bin",
        .default_output = "bin/f5-2_y.bin",
        .suite = FEISTEL,
        .params.feistel_params = {.iter = 2, .shift = 5},
    },
    {
        .name = "F6-2",
        .gen_model_func = feistel_generator,
        .description = "Use feistel model with 6 shift",
        .default_input = "bin/f6-2_x.bin",
        .default_output = "bin/f6-2_y.bin",
        .suite = FEISTEL,
        .params.feistel_params = {.iter = 2, .shift = 6},
    },
    {
        .name = "F7-2",
        .gen_model_func = feistel_generator,
        .description = "Use feistel model with 7 shift",
        .default_input = "bin/f7-2_x.bin",
        .default_output = "bin/f7-2_y.bin",
        .suite = FEISTEL,
        .params.feistel_params = {.iter = 2, .shift = 7},
    },
};

void print_models();
void print_generators();

generator_type_t *get_type_by_name(const char *name, generator_type_t *list, uint8_t size);

generator_type_t *get_generator_by_name(const char *name);
model_type_t *get_model_by_name(const char *name);

#endif //GENERATOR_H_