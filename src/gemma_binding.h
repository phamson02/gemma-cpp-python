#include "gemma.h" // Gemma
#include "util/args.h"
#include <iostream>

#pragma once
class GemmaModel {
    private:
        gcpp::Gemma *model;
        gcpp::Model model_type;

        const int eos_token = 1;
        const int bos_token = 2;

    public:
        GemmaModel(const char *tokenizer_path_str, const char *compressed_weights_path_str,
                    int model_type_id);

        ~GemmaModel();

        int get_bos_token() const;

        int get_eos_token() const;

        std::vector<int> tokenize(const std::string &text, bool add_bos = true);

        std::string detokenize(const std::vector<int> &tokens);

        std::string complete(const std::string &text);
};