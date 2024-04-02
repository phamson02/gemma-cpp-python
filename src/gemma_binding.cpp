#include "gemma.h" // Gemma
#include "util/args.h"
#include <iostream>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

gcpp::Path createPath(const char *pathString) {
    gcpp::Path path;
    path = pathString;
    return path;
}

class _GemmaModel {
    private:
        gcpp::Gemma *model;
        gcpp::Model model_type;

        const int eos_token = 1;
        const int bos_token = 2;

    public:
        _GemmaModel(const char *tokenizer_path_str, const char *compressed_weights_path_str,
                    int model_type_id) {
            gcpp::Path tokenizer_path = createPath(tokenizer_path_str);
            gcpp::Path compressed_weights_path = createPath(compressed_weights_path_str);
            model_type = static_cast<gcpp::Model>(model_type_id);

            // Rough heuristic for the number of threads to use
            size_t num_threads = static_cast<size_t>(std::clamp(
                                     static_cast<int>(std::thread::hardware_concurrency()) - 2, 1, 18));
            hwy::ThreadPool pool(num_threads);

            model = new gcpp::Gemma(tokenizer_path, compressed_weights_path, model_type, pool);
        }

        ~_GemmaModel() {
            delete model;
        }

        int get_bos_token() const {
            return bos_token;
        }

        int get_eos_token() const {
            return eos_token;
        }

        std::vector<int> tokenize(const std::string &text, bool add_bos = true) {
            std::vector<int> tokens;

            if (model->Tokenizer()->Encode(text, &tokens).ok()) {
                if (add_bos) {
                    tokens.insert(tokens.begin(), bos_token);
                }
            }

            return tokens;
        }

        std::string detokenize(const std::vector<int> &tokens) {
            std::string text;
            if (model->Tokenizer()->Decode(tokens, &text).ok()) {
                return text;
            }
            return "";
        }

        std::string complete(const char *text) {
            // Rough heuristic for the number of threads to use
            size_t num_threads = static_cast<size_t>(std::clamp(
                static_cast<int>(std::thread::hardware_concurrency()) - 2, 1, 18));
            hwy::ThreadPool pool(num_threads);

            auto kv_cache = CreateKVCache(model_type);
            size_t pos = 0; // KV Cache position

            // Initialize random number generator
            std::mt19937 gen;
            std::random_device rd;
            gen.seed(rd());

            std::vector<int> tokens = tokenize(text);
            size_t ntokens = tokens.size();

            std::string completion;

            // This callback function gets invoked everytime a token is generated
            auto stream_token = [this, &pos, &gen, &ntokens, tokenizer = model->Tokenizer(), &completion] (
                                    int token, float) {
                ++pos;
                if (pos < ntokens) {
                    // print feedback
                }
                else if (token != this->eos_token) {
                    std::string token_text;
                    HWY_ASSERT(tokenizer->Decode(std::vector<int>{token}, &token_text).ok());

                    // Append token to the completion string
                    completion += token_text;
                }
                return true;
            };

            gcpp::GenerateGemma(*model,
                          {.max_tokens = 2048,
                           .max_generated_tokens = 1024,
                           .temperature = 0.0,
                           .verbosity = 0},
                          tokens, /*KV cache position = */ 0, kv_cache, pool,
                          stream_token, gen);

            return completion;
        }
};

PYBIND11_MODULE(_pygemma, m) {
    m.doc() = "Python binding for gemma.cpp";

    py::class_<_GemmaModel>(m, "_GemmaModel")
    .def_property_readonly("bos_token", &_GemmaModel::get_bos_token, "Get the BOS token")
    .def(py::init<const char *, const char *, int>(), py::arg("tokenizer_path"),
         py::arg("compressed_weights_path"), py::arg("model_type"), "Create an instance of Gemma model")
    .def("tokenize", &_GemmaModel::tokenize, py::arg("text"), py::arg("add_bos") = false,
         "Tokenize the input text and return the tokenized text")
    .def("detokenize", &_GemmaModel::detokenize, py::arg("tokens"),
         "Detokenize the input tokens and return the detokenized text")

    // Quick experiment
    .def("complete", &_GemmaModel::complete, py::arg("text"),
         "Complete the input text and return the completion text");
}