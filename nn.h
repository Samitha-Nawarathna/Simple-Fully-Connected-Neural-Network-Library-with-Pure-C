#ifndef NN_H_
#define NN_H_

#include <stdio.h>
#include <stdbool.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>
#include "./matrix.h"

//structures definitions

enum ParamType {weight, bias};
enum Activation {linear, relu, sigmoid, softmax, tanha};
#define LEN(arr) (sizeof((arr)) / sizeof((arr[0])))
// #define PARAM(model, layer, p, i, j) MAT_AT((p == 0) model.layers[layer].weight: model.layers[layer].bias, i, j)

typedef struct 
{
    MAT weight;
    MAT bias;
    enum Activation activation;

}Linear;

typedef struct 
{
    size_t num_layers;
    Linear *layers;
    size_t (*arch)[3];
}Model;


//Activation function definitions

void nn_softmaxf(MAT *dst, MAT *mat);

float nn_linearf(float x);
float nn_reluf(float x);
float nn_sigmoidf(float x);
float nn_tanhf(float x);
float squaref(float x);


//core functionalities definitions

Linear nn_linear_alloc(size_t input_shape, size_t output_shape, enum Activation activation);
MAT nn_linear_forward(Linear *linear, MAT *x);
void nn_layer_print(Linear layer);
Model nn_model_alloc(size_t layer_count, size_t arch[][3]);
MAT nn_model_forward(Model *model, MAT *x);
Model nn_model_clone(Model *model);
void nn_model_print(Model model);
float *nn_model_param(Model *model, size_t layer, enum ParamType param_type, size_t i, size_t j);
void nn_dealloc(Model *model);

float calc_cost(Model *model, MAT *x, MAT *y);
float calc_loss(Model *model, MAT *dataset);
float calc_grad_param(Model *model, Model *model_step, float *param,  MAT *dataset, float eps);
Model calc_grad_model(Model *model, MAT *dataset, float eps);
#endif //NN_H_

#ifndef NN_IMPLEMENTATION
#define NN_IMPLEMENTATION

//core functionalities implementations

Linear nn_linear_alloc(size_t input_shape, size_t output_shape, enum Activation activation)
{
    Linear linear;
    linear.weight = mat_alloc(input_shape, output_shape);
    linear.bias =  mat_alloc(1, output_shape);
    linear.activation = activation;

    return linear;
}

MAT nn_linear_forward(Linear *linear, MAT *x)
{
    // printf("%d\n", linear->bias.cols);
    // nn_print_layer(*linear);
    MAT output = mat_alloc(linear->bias.cols, linear->bias.rows);
    mat_dot(&output, &linear->weight, x);
    mat_add(&output, &linear->bias, &output);

    switch (linear->activation)
    {
    case softmax:
        nn_softmaxf(&output, &output);
        break;
    case relu:
        mat_apply(&output, &output, nn_reluf);    
        break;
    case sigmoid:
        mat_apply(&output, &output, nn_sigmoidf);    
        break;
    case tanha:
        mat_apply(&output, &output, nn_tanhf);    
        break;                
    default:
        mat_apply(&output, &output, nn_linearf);
        break;
    }


    return output;
}

void nn_layer_print(Linear layer)
{
    printf("weights = ");
    mat_print(layer.weight);
    // printf("%d\n", layer.bias.rows);
    printf("bias = ");
    mat_print(layer.bias);
}

Model nn_model_alloc(size_t layer_count, size_t arch[][3])
{
    Model model;
    model.num_layers = layer_count;
    model.layers = (Linear* )malloc(sizeof(Linear)*layer_count);
    model.arch = arch;

    for (size_t i = 0; i < layer_count; i++)
    {
        Linear layer = nn_linear_alloc(arch[i][0], arch[i][1], arch[i][2]); 
        model.layers[i] = layer;
    }
    return model;
}

void nn_model_print(Model model)
{
    for (size_t i = 0; i < model.num_layers; i++)
    {
        printf("layer %d:\n", i);
        nn_layer_print(model.layers[i]);
        printf("\n\n");
    }
}

MAT nn_model_forward(Model *model, MAT *x)
{
    MAT temp = *x;

    for (size_t i = 0; i < model->num_layers; i++)
    {
        temp = nn_linear_forward(&model->layers[i], &temp);
    }
    return temp;
}

Model nn_model_clone(Model *model)
{
    Model model_clone = nn_model_alloc(model->num_layers, model->arch);

    for (size_t layer = 0; layer < model->num_layers; layer++)
    {
        for (size_t paramtype = 0; paramtype < 2; paramtype++)
        {
            MAT *param_org = (paramtype == 0) ? &model->layers[layer].weight : &model->layers[layer].bias;
            MAT *param_clone = (paramtype == 0) ? &model_clone.layers[layer].weight : &model_clone.layers[layer].bias;

            mat_clone(param_clone, param_org);
        }
    }
    return model_clone;        

}

float *nn_model_param(Model *model, size_t layer, enum ParamType param_type, size_t i, size_t j)
{

    assert(layer < model->num_layers);

    MAT temp;
    switch (param_type)
    {
    case weight:
        temp = model->layers[layer].weight;
        break;
    
    case bias:
        temp = model->layers[layer].bias;
        break;

    default:
        printf("invalid Parameter");
        break;    
    }

    assert(temp.rows > i);
    assert(temp.cols > j);

    return &MAT_AT(temp, i, j);
}

void nn_dealloc(Model *model)
{
    for (size_t layer = 0; layer < model->num_layers; layer++)
    {
        for (size_t paramtype = 0; paramtype < 2; paramtype++)
        {
            MAT *param_org = (paramtype == 0) ? &model->layers[layer].weight : &model->layers[layer].bias;
            mat_dealloc(param_org);

        }
    }
}



//activation functions implementations

void nn_softmaxf(MAT *dst, MAT *mat)
{
    mat_apply(dst, mat, expf);
    mat_scale(dst, mat, 1/mat_sum(mat));
}

float nn_linearf(float x)
{
    return x;
}

float nn_reluf(float x)
{
    return (x > 0) ? x : 0;
}

float nn_sigmoidf(float x)
{
    return 1/(1 + expf(-x));
}

float nn_tanhf(float x)
{
    return (expf(x) - expf(-x))/(expf(x) + expf(-x));
}

float squaref(float x)
{
    return x*x;
}

float calc_cost(Model *model, MAT *x, MAT *y)
{
    // printf("x_shape: (%d, %d)\n", x->rows, x->cols);
    // printf("y_shape: (%d, %d)\n", y->rows, y->cols);

    MAT pred = nn_model_forward(model, x);
    // mat_print(pred);
    float cost;
    mat_scale(y, y, -1);
    mat_add(&pred, &pred, y);
    // mat_print(pred);
    // printf("added!\n");
    mat_apply(&pred, &pred, squaref);
    cost = mat_sum(&pred);

    // printf("diff = %f\n", cost);
    return cost;
}


float calc_loss(Model *model, MAT *dataset)
{
    float loss = 0;
    MAT dataset_transposed = mat_transpose(dataset);
    // printf("%d\n", dataset_transposed.cols);
    // printf("%d\n", dataset_transposed.rows);
    for (size_t i = 0; i < dataset_transposed.cols; i++)
    {
        MAT x = mat_select(&dataset_transposed, 0, dataset_transposed.rows - model->layers[model->num_layers - 1].weight.rows, i, 1);
        MAT y = mat_select(&dataset_transposed, model->layers[model->num_layers - 1].weight.rows + 1, model->layers[model->num_layers - 1].weight.rows , i, 1);
        loss += calc_cost(model, &x, &y);
    }
    // mat_dealloc(&dataset_transposed);
    return loss / dataset->rows;
}

float calc_grad_param(Model *model, Model *model_step, float *param,  MAT *dataset, float eps)
{
    // nn_model_print(*model);
    
    float grad;
    *param += eps;
    // nn_model_print(*model_step);
    grad = (calc_loss(model_step, dataset) - calc_loss(model, dataset)) / eps;
    // printf("grad : %f\n", grad);
    return grad;

}

Model calc_grad_model(Model *model, MAT *dataset, float eps)
{
    Model model_grad = nn_model_alloc(model->num_layers, model->arch);

    nn_model_print(model_grad);
    for (size_t layer = 0; layer < model->num_layers; layer++)
    {
        for (size_t paramtype = 0; paramtype < 2; paramtype++)
        {
            MAT param = (paramtype == 0) ? model->layers[layer].weight : model->layers[layer].bias;
            for (size_t i = 0; i < param.rows; i++)
            {
                for (size_t j = 0; j < param.cols; j++)
                {
                    Model model_step = nn_model_clone(model);
                    float *param_model_step = nn_model_param(&model_step, layer, paramtype, i, j);
                    float *param_model_grad = nn_model_param(&model_grad, layer, paramtype, i, j);

                    *param_model_grad = calc_grad_param(model, &model_step, param_model_step, dataset, eps);
                }
                
            }
        }
    }
    return model_grad;
}

void train_step(Model *model, MAT *dataset, float eps, float lr)
{

    for (size_t layer = 0; layer < model->num_layers; layer++)
    {
        for (size_t paramtype = 0; paramtype < 2; paramtype++)
        {
            MAT param = (paramtype == 0) ? model->layers[layer].weight : model->layers[layer].bias;
            for (size_t i = 0; i < param.rows; i++)
            {
                for (size_t j = 0; j < param.cols; j++)
                {
                    Model model_step = nn_model_clone(model);
                    float *param_model_step = nn_model_param(&model_step, layer, paramtype, i, j);
                    float *param_model = nn_model_param(model, layer, paramtype, i, j);

                    *param_model += -1*calc_grad_param(model, &model_step, param_model_step, dataset, eps)*lr;
                    // nn_dealloc(&model_step);
                }
                
            }
        }
    }
}

void train_loop(float iter, Model *model, MAT *dataset, float eps, float lr)
{
    for (size_t i = 0; i < iter; i++)
    {
        train_step(model, dataset, eps, lr);
        printf("Iter: %d, Train Loss: %f\n", i, calc_loss(model, dataset));
    }
    
}





#endif //NN_IMPLEMENTATION