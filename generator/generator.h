#ifndef GENERATOR_H_
#define GENERATOR_H_

#include <string.h>
#include <stdlib.h>
#include <time.h>
#include <inttypes.h>
#include <stdio.h>

#include "crypto.h"

typedef struct generator_params_t generator_params_t;
typedef void (*generator)(crypto_tfm *ctx, generator_params_t *params, uint64_t size,
                          const char *filename_x, const char *filename_y,
                          void (*generator)(crypto_tfm *ctx, FILE *out_file_x, FILE *out_file_y, uint64_t in));

typedef void (*generator_model)(crypto_tfm *ctx, FILE *out_file_x, FILE *out_file_y, uint64_t in);

/**
 * @brief Iterate from 0 to @size as input to @gen function
 * 
 * @param ctx           Crypto context
 * @param params        Params for generator, can be NULL
 * @param size          size of output sequence <= UINT64_MAX
 * @param filename_x    filename with input
 * @param filename_y    filename with output
 * @param gen           generator of output sequence
 */
void iterate_generator(crypto_tfm *ctx, generator_params_t *params,
                       uint64_t size, const char *filename_x, const char *filename_y,
                       generator_model gen);

/**
 * @brief Iterate from 0 to @size in two 32-bite blocks as input to @gen function
 * 
 * @param ctx           Crypto context
 * @param params        Params for generator, can be NULL
 * @param size          size of output sequence <= UINT32_MAX
 * @param filename_x    filename with input
 * @param filename_y    filename with output
 * @param gen           generator of output sequence
 */
void iterate_parallel_generator(crypto_tfm *ctx, generator_params_t *params,
                                uint64_t size, const char *filename_x, const char *filename_y,
                                generator_model gen);

/**
 * @brief Iterate from 0 to @size in two <S>-bite blocks as input to @gen function.
 *        <S> from params. Support <S> = {4, 8, 16, 32}. Size of output is @size^2.
 * 
 * @param ctx           Crypto context
 * @param params        Params for generator, can be NULL
 * @param size          size of output sequence <= UINT32_MAX
 * @param filename_x    filename with input
 * @param filename_y    filename with output
 * @param gen           generator of output sequence
 */
void iterate_split_generator(crypto_tfm *ctx, generator_params_t *params,
                             uint64_t size, const char *filename_x, const char *filename_y,
                             generator_model gen);

/**
 * @brief Generate random @size values as input to @gen function
 * 
 * @param ctx           Crypto context
 * @param params        Params for generator, can be NULL
 * @param size          size of output sequence <= UINT64_MAX
 * @param filename_x    filename with input
 * @param filename_y    filename with output
 * @param gen           generator of output sequence
 */
void random_generator(crypto_tfm *ctx, generator_params_t *params,
                      uint64_t size, const char *filename_x, const char *filename_y,
                      generator_model gen);

/**
 * @brief Iterate values from 0 to @size in random order as input to @gen function
 * 
 * @param ctx           Crypto context
 * @param params        Params for generator, can be NULL
 * @param size          size of output sequence <= UINT64_MAX
 * @param filename_x    filename with input
 * @param filename_y    filename with output
 * @param gen           generator of output sequence
 */
void random_iterate_generator(crypto_tfm *ctx, generator_params_t *params,
                              uint64_t size, const char *filename_x, const char *filename_y,
                              generator_model gen);

/**
 * @brief Generator for 1-round of GOST
 * 
 * @param ctx           Crypto context
 * @param out_file_x    File context to write input
 * @param out_file_y    File context to write output
 * @param in            Input value
 */
void round_generator(crypto_tfm *ctx, FILE *out_file_x, FILE *out_file_y, uint64_t in);

/**
 * @brief Generator for 2-round of GOST
 * 
 * @param ctx           Crypto context
 * @param out_file_x    File context to write input
 * @param out_file_y    File context to write output
 * @param in            Input value
 */
void n_round_generator(crypto_tfm *ctx, FILE *out_file_x, FILE *out_file_y, uint64_t in);

/**
 * @brief Generator for G0 model. Input - 8 bits, output - 4 bits.
 * 
 * @param ctx           Crypto context
 * @param out_file_x    File context to write input
 * @param out_file_y    File context to write output
 * @param in            Input value
 */
void primitive_g0_generator(crypto_tfm *ctx, FILE *out_file_x, FILE *out_file_y, uint64_t in);

/**
 * @brief Generator for G1 model. Input - 8 bits, output - 4 bits.
 * 
 * @param ctx           Crypto context
 * @param out_file_x    File context to write input
 * @param out_file_y    File context to write output
 * @param in            Input value
 */
void primitive_g1_generator(crypto_tfm *ctx, FILE *out_file_x, FILE *out_file_y, uint64_t in);

/**
 * @brief Generator for G2 model. Input - 8 bits, output - 4 bits.
 * 
 * @param ctx           Crypto context
 * @param out_file_x    File context to write input
 * @param out_file_y    File context to write output
 * @param in            Input value
 */
void primitive_g2_generator(crypto_tfm *ctx, FILE *out_file_x, FILE *out_file_y, uint64_t in);

/**
 * @brief Generator for G3 model. Input - 64 bits, output - 32 bits.
 * 
 * @param ctx           Crypto context
 * @param out_file_x    File context to write input
 * @param out_file_y    File context to write output
 * @param in            Input value
 */
void primitive_g3_generator(crypto_tfm *ctx, FILE *out_file_x, FILE *out_file_y, uint64_t in);

/**
 * @brief Generator for G4 model. 
 * Param in ctx shows (@input) number of bits in input.
 * Support next @input: 4 bits  -> 4 bits
 *                      8 bits  -> 8 bits
 *                      16 bits -> 16 bits
 *                      32 bits -> 32 bits
 * 
 * @param ctx           Crypto context
 * @param out_file_x    File context to write input
 * @param out_file_y    File context to write output
 * @param in            Input value
 */
void primitive_g4_generator(crypto_tfm *ctx, FILE *out_file_x, FILE *out_file_y, uint64_t in);

/**
 * @brief Generator for Feistel model. Input - 16 bits, output - 16 bits.
 * 
 * @param ctx           Crypto context
 * @param out_file_x    File context to write input
 * @param out_file_y    File context to write output
 * @param in            Input value
 */
void feistel_generator(crypto_tfm *ctx, FILE *out_file_x, FILE *out_file_y, uint64_t in);

void generate_random_key(uint8_t *key);

typedef struct generator_type_t generator_type_t;
typedef struct model_type_t model_type_t;

typedef model_type_t *(*model_formatter)(const char *);
typedef generator_type_t *(*generator_formatter)(const char *);

/**
 * Struct with input and output params of generator
 */
struct generator_params_t
{
    /* Split input  */
    uint8_t split;
    /* Size left input, if split == 0, then left is main */
    uint8_t left_input;
    /* Size right input, if split == 0, then right is 0 */
    uint8_t right_input;
};

/**
 * Struct with generator's description 
 */
struct generator_type_t
{
    /* Generator name */
    char *name;
    /* Description of generator */
    char *description;
    /* Pointer to generator */
    generator gen_func;
    /* Params for generator */
    generator_params_t params;
    /* Generator formatter for specific generator that needs param */
    generator_formatter formatter;
};

/**
 * Struct with model's description
 */
struct model_type_t
{
    /* Model name */
    char *name;
    /* Model description */
    char *description;
    /* Default filepath for output */
    char *default_input;
    /* Default filepath for input */
    char *default_output;
    /* Pointer to model generator */
    generator_model gen_model_func;
    /* Model ciphersuite */
    ciphersuite_t suite;
    /* Params for model */
    crypto_params params;
    /* Model formatter for model that needs params */
    model_formatter formatter;
};

/**
 * @brief Formatter for Feistel models
 * 
 * @param str               input string
 *  
 * @return model_type_t     NULL if str is incorrect 
 */
model_type_t *feistel_formatter(const char *str);

/**
 * @brief Formatter for G4's model
 * 
 * @param str               input string
 *  
 * @return model_type_t     NULL if str is incorrect 
 */
model_type_t *g4_formatter(const char *str);

/**
 * @brief Formatter for iter-split generator
 * 
 * @param str                   input string
 *  
 * @return generator_type_t     NULL if str is incorrect 
 */
generator_type_t *iter_split_formatter(const char *str);

/* List of generators */
static generator_type_t generators[] = {
    {
        .name = "iter",
        .gen_func = iterate_generator,
        .description = "Iterate from 0 to size",
    },
    {
        .name = "iter2",
        .gen_func = iterate_parallel_generator,
        .description = "Iterate from 0 to size in 2 32-bite blocks",
    },
    {
        .name = "rand",
        .gen_func = random_generator,
        .description = "Generate random value as input",
    },
    {
        .name = "rand2",
        .gen_func = random_iterate_generator,
        .description = "Iterate from 0 to size in random order",
    },
    {
        .name = "iter-split-<S>",
        .gen_func = iterate_split_generator,
        .description = "Iterate from 0 to size. Input=Left|Right. <S> - size of one part in bits",
        .formatter = iter_split_formatter,
    },
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
        .name = "G4-<I>",
        .formatter = g4_formatter,
        .gen_model_func = primitive_g4_generator,
        .description = "Use G4 model with <I> inputs",
        .default_input = "bin/g4-<I>_x.bin",
        .default_output = "bin/g4-<I>_y.bin",
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

void print_models();
void print_generators();

generator_type_t *get_type_by_name(const char *name, generator_type_t *list, uint8_t size);

generator_type_t *get_generator_by_name(const char *name);
model_type_t *get_model_by_name(const char *name);

void destroy_model(model_type_t *model);
void destroy_generator(generator_type_t *generator);

#endif //GENERATOR_H_