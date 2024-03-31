#include "gemma.h" // Gemma
#include "util/args.h"
#include <iostream>

gcpp::Path createPath(const char *pathString)
{
    gcpp::Path path;
    path = pathString;
    return path;
}

extern "C" {

gcpp::Gemma *loadGemmaModel(const char *tokenizer_path_str, const char *compressed_weights_path_str, const char *model_type_str)
{
    gcpp::Path tokenizer_path = createPath(tokenizer_path_str);
    gcpp::Path compressed_weights_path = createPath(compressed_weights_path_str);

    gcpp::Model model_type;
    if (std::string(model_type_str) == "2b-pt" || std::string(model_type_str) == "2b-it") {
        model_type = gcpp::Model::GEMMA_2B;
    }
    else {
        model_type = gcpp::Model::GEMMA_7B;
    }

    // Rough heuristic for the number of threads to use
    size_t num_threads = static_cast<size_t>(std::clamp(
        static_cast<int>(std::thread::hardware_concurrency()) - 2, 1, 18));
    hwy::ThreadPool pool(num_threads);

    gcpp::Gemma *model = new gcpp::Gemma(tokenizer_path, compressed_weights_path, model_type, pool);
    return model;
}
}