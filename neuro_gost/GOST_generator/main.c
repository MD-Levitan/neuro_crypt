#include "generator.h"

const char *usage = "Usage:  ./generator <SIZE> <GEN_TYPE> <MODEL> [INPUT] [OUTPUT]";

/*
 *	Generator
 *  Usage:  ./generator <SIZE> <GEN_TYPE> <MODEL>
 *
 */
int main(int argc, const char **argv)
{
    if (argc < 4)
    {
        printf("%s!\n", usage);
        print_generators();
        print_models();
        return -1;
    }

    int8_t key[32];
    const char *model_name, *gen_name;
    const char *input_file = NULL, *output_file = NULL;
    uint64_t size = 0;
    generator_type_t *model, *generator;

    struct crypto_tfm *ctx;

    size = strtoll(argv[1], NULL, 10);
    gen_name = argv[2];
    model_name = argv[3];

    if (argc >= 5)
    {
        input_file = argv[4];
        output_file = argv[5];
    }

    ctx = create_tfm_ctx();
    generate_random_key(key);
    magma_setkey(ctx, key, sizeof(key));

    model = get_model_by_name(model_name);
    generator = get_generator_by_name(gen_name);

    if (!model || !generator)
    {
        printf("Incorrect model or generator!\n");
        printf("%s!\n", usage);
        print_generators();
        print_models();
        delete_tfm_ctx(ctx);
        return -1;
    }
    input_file = input_file ? input_file : model->default_input;
    output_file = output_file ? output_file : model->default_output;

    printf("Generate sequence with following params:\n\tgenerator - %s\n\tmodel - %s\n\tsize - %lld\n\tinput - %s\n\toutput - %s\n",
           generator->name, model->name, size, input_file, output_file);
    generator->func.gen_func(ctx, size, input_file, output_file, model->func.gen_model_func);

    delete_tfm_ctx(ctx);
}