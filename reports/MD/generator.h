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
 * @brief Generator for G5 model. Input - 4 bits, output - 4 bits.
 * 
 * @param ctx           Crypto context
 * @param out_file_x    File context to write input
 * @param out_file_y    File context to write output
 * @param in            Input value
 */
void primitive_g5_generator(crypto_tfm *ctx, FILE *out_file_x, FILE *out_file_y, uint64_t in);


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

