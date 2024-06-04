#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include ".\matrix.h"
#include ".\nn.h"
// #include ".\dataframe.h"

int main()
{

    // float data[][3] = {
    //     {0, 0, 5},
    //     {0, 1, 7},
    //     {0, 2, 9},
    //     {1, 0, 5},
    //     {1, 1, 7},
    //     {1, 2, 9},
    //     {2, 0, 5},
    //     {2, 1, 7},
    //     {2, 2, 9},

    // };
    FILE *file;
    file = fopen(".\\data\\train.csv", "r");
    size_t i = 0, rows = 5, cols = 785;
    char line[10*1000];
    MAT dataset = mat_alloc(cols, rows);

    while((fgets(line, sizeof(line), file) != NULL && i < rows))
    {
        if (i != 0)
        {
            size_t j = 0;
        
            char *token = strtok(line, ",");
            while (token != 0)
            {
                
                float value = atof(token);
                MAT_AT(dataset, i, j) = value;
                // printf("%f\n", value);
                // printf("(%d, %d)\n", i, j);

                token = strtok(0, ",");
                j+=1;
            }
            // printf("(%d, %d)\n", i, j);
        }
        
        i+=1;
    }

    printf("Dataset loaded to memory!"); 

    // MAT dataset;
    // dataset.cols = 3;
    // dataset.rows = LEN(data);
    // dataset.elements = (float*)&data[0][0];


    size_t arch[][3] = {
        {28*28, 28, relu},
        {28, 28, relu},
        {28, 10, relu},
        {10, 1, sigmoid}
    };

    // MAT input = mat_alloc(1, 2);
    // mat_fill(&input, 2);
    Model model = nn_model_alloc(LEN(arch), arch);
    // for (size_t i = 0; i < model.num_layers; i++)
    // {
    //     mat_fill(&model.layers[i].weight, 0);
    //     mat_fill(&model.layers[i].bias, 0);
    // }
    // *nn_model_param(&model, 0, weight, 0, 0) = 0;
    // *nn_model_param(&model, 0, weight, 0, 1) = 0;
    // *nn_model_param(&model, 0, bias, 0, 0) = 0;
    // float loss = calc_loss(&model, &dataset);


    nn_model_print(model);
    train_loop(1, &model, &dataset, 1e-3, 1e-2);
    nn_model_print(model);
    // MAT output = nn_model_forward(&model, &input);
    // mat_print(output);
    // printf("PARAM : %f\n", *nn_model_param(&model, 0, bias, 0, 0));
    // printf("loss: %f", loss);



    return 0;
}