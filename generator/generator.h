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
 * @brief Generator for G4 model. Input - 4 bits, output - 4 bits.
 * 
 * @param ctx           Magma context
 * @param out_file_x    File context to write input
 * @param out_file_y    File context to write output
 * @param in            Input value
 */
void primitive_g4_generator(crypto_tfm *ctx, FILE *out_file_x, FILE *out_file_y, uint64_t in);

/**
 * @brief Generator for G4-Long model. Input - 8 bits, output - 8 bits.
 * 
 * @param ctx           Magma context
 * @param out_file_x    File context to write input
 * @param out_file_y    File context to write output
 * @param in            Input value
 */
void primitive_g4l_generator(crypto_tfm *ctx, FILE *out_file_x, FILE *out_file_y, uint64_t in);

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
typedef model_type_t * (*formatter)(const char *);

/**
 * @brief Formatter for Feistel models
 * 
 * @param str               input string
 *  
 * @return model_type_t     NULL if str is incorrect 
 */
model_type_t *feistel_formatter(const char *str);

struct generator_type_t
{
    const char *name;
    generator gen_func;
    const char *description;
};

struct model_type_t
{
    char *name;
    formatter formatter;
    generator_model gen_model_func;
    char *description;
    char *default_input;
    char *default_output;
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
        .formatter = NULL,
        .gen_model_func = round_generator,
        .description = "Use 1-round GOST encryption",
        .default_input = "bin/n1_x.bin",
        .default_output = "bin/n1_y.bin",
        .suite = MAGMA,
    },
    {
        .name = "GOST-2",
        .formatter = NULL,
        .gen_model_func = n_round_generator,
        .description = "Use 2-round GOST encryption",
        .default_input = "bin/n2_x.bin",
        .default_output = "bin/n2_y.bin",
        .suite = MAGMA,
    },
    {
        .name = "G0",
        .formatter = NULL,
        .gen_model_func = primitive_g0_generator,
        .description = "Use G0 model",
        .default_input = "bin/g0_x.bin",
        .default_output = "bin/g0_y.bin",
        .suite = MAGMA,
    },
    {
        .name = "G1",
        .formatter = NULL,
        .gen_model_func = primitive_g1_generator,
        .description = "Use G1 model",
        .default_input = "bin/g1_x.bin",
        .default_output = "bin/g1_y.bin",
        .suite = MAGMA,
    },
    {
        .name = "G2",
        .formatter = NULL,
        .gen_model_func = primitive_g2_generator,
        .description = "Use G2 model",
        .default_input = "bin/g2_x.bin",
        .default_output = "bin/g2_y.bin",
        .suite = MAGMA,
    },
    {
        .name = "G3",
        .formatter = NULL,
        .gen_model_func = primitive_g3_generator,
        .description = "Use G3 model",
        .default_input = "bin/g3_x.bin",
        .default_output = "bin/g3_y.bin",
        .suite = MAGMA,
    },
    {
        .name = "G4",
        .formatter = NULL,
        .gen_model_func = primitive_g4_generator,
        .description = "Use G4 model",
        .default_input = "bin/g4_x.bin",
        .default_output = "bin/g4_y.bin",
        .suite = MAGMA,
    },
    {
        .name = "G4L",
        .formatter = NULL,
        .gen_model_func = primitive_g4l_generator,
        .description = "Use G4-Long model",
        .default_input = "bin/g4l_x.bin",
        .default_output = "bin/g4l_y.bin",
        .suite = MAGMA,
    },
    {
        .name = "F<I>-<S>",
        .formatter = feistel_formatter,
        .gen_model_func = feistel_generator,
        .description = "Use feistel model with <S> shift and <I> iteration",
        .default_input = "bin/f<I>-<S>_x.bin",
        .default_output = "bin/f<I>-<S>_y.bin",
        .suite = FEISTEL,
        .params.feistel_params = {.iter = 1, .shift = 1},
    },
};

void destroy_model(model_type_t *model);
void print_models();
void print_generators();

generator_type_t *get_type_by_name(const char *name, generator_type_t *list, uint8_t size);

generator_type_t *get_generator_by_name(const char *name);
model_type_t *get_model_by_name(const char *name);

#endif //GENERATOR_H_