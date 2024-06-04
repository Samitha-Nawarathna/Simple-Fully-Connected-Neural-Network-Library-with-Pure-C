#ifndef MAT_H_
#define MAT_H_

#include <stdio.h>
#include <stdbool.h>
#include <stdlib.h>
#include <assert.h>

typedef struct 
{
    size_t rows;
    size_t cols;
    float *elements;

} MAT;

#define MAT_AT(m, i, j) (m).elements[(i)*(m).cols + (j)]

MAT mat_alloc(size_t cols, size_t rows);
void mat_fill(MAT *mat, float value);
void mat_print(MAT mat);
void mat_clone(MAT *dst, MAT *mat);
void mat_dealloc(MAT *mat);
MAT mat_transpose(MAT *mat);
void mat_add(MAT *dst, MAT *a, MAT *b);
void mat_mul(MAT *dst, MAT *a, MAT *b);
void mat_dot(MAT *dst, MAT *a, MAT *b);
float mat_sum(MAT *mat);
void mat_scale(MAT *dst, MAT *mat, float value);
void mat_apply(MAT *dst, MAT *mat, float (*function)(float));
MAT mat_select(MAT *parent, size_t row_start, size_t row_count, size_t col_start, size_t col_count);
// static void mat_iter(MAT *mat, );


#endif //MAT_H

#ifndef MAT_IMPLEMENTATION
#define MAT_IMPLEMENTATION 

MAT mat_alloc(size_t cols, size_t rows)
{
    MAT mat;
    mat.cols = cols;
    mat.rows = rows;
    mat.elements = malloc(sizeof(float)*mat.cols*mat.rows);

    return mat;
}

void mat_fill(MAT *mat, float value)
{
    for (size_t i=0; i < mat->rows; i++)
    {
        for (size_t j=0; j < mat->cols; j++)
        {
            MAT_AT(*mat, i, j) = value;
        }
    }      
}

void mat_print(MAT mat)
{
    printf("[\n");
    for (size_t i=0; i < mat.rows; i++)
    {
        for (size_t j=0; j < mat.cols; j++)
        {
            printf("%f ", MAT_AT(mat, i, j));
        }
        printf("\n");
    }
    printf("]\n");
}

void mat_clone(MAT *dst, MAT *mat)
{
    for (size_t i = 0; i < mat->rows; i++)
    {
        for (size_t j = 0; j < mat->cols; j++)
        {
            MAT_AT(*dst, i, j) = MAT_AT(*mat, i, j);
        }
    }
}

void mat_dealloc(MAT *mat)
{
    for (size_t i = 0; i < mat->rows; i++)
    {
        for (size_t j = 0; j < mat->cols; j++)
        {
            for (size_t k = 0; k < mat->cols*mat->cols; k++)
            {
                free(&mat->elements[k]);
            }
            free(mat->elements);
            
        }
    }
}


MAT mat_transpose(MAT *mat)
{
    MAT mat_tranposed = mat_alloc(mat->rows, mat->cols);
    
    for (size_t i = 0; i < mat->rows; i++)
    {
        for (size_t j = 0; j < mat->cols; j++)
        {
            MAT_AT(mat_tranposed, j, i) = MAT_AT(*mat, i, j);
        }
            
    }
    // mat_print(mat_tranposed);
    return mat_tranposed; 
    
}


void mat_add(MAT *dst, MAT *a, MAT *b)
{
    // mat_print(*b);
    assert (a->cols == b->cols);
    assert (a->rows == b->rows);
    assert (a->cols == dst->cols);
    assert (a->rows == dst->rows);
    

    for (size_t i=0; i < dst->rows; i++)
    {
        for (size_t j=0; j < dst->cols; j++)
        {
            MAT_AT(*dst, i, j) = MAT_AT(*a, i, j) + MAT_AT(*b, i, j);
        }
    }

}

void mat_dot(MAT *dst, MAT *a, MAT *b)
{
    assert (a->cols == b->rows);
    assert (dst->rows == a->rows);
    assert(dst->cols == b->cols);

    for (size_t i=0; i < dst->rows; i++)
    {
        for (size_t j=0; j < dst->cols; j++)
        {
            MAT_AT(*dst, i, j) = 0;

            for (size_t k=0; k < a->cols; k++)
            {
                // printf("%f\n", MAT_AT(*a, i, k)*MAT_AT(*b, k, j));
                MAT_AT(*dst, i, j) += MAT_AT(*a, i, k)*MAT_AT(*b, k, j);
            }
        }
    }

}

void mat_mul(MAT *dst, MAT *a, MAT *b)
{
    assert (a->cols == b->cols);
    assert (a->rows == b->rows);
    assert (a->cols == dst->cols);
    assert (a->rows == dst->rows);
    

    for (size_t i=0; i < dst->rows; i++)
    {
        for (size_t j=0; j < dst->cols; j++)
        {
            MAT_AT(*dst, i, j) = MAT_AT(*a, i, j)*MAT_AT(*b, i, j);
        }
    }
}

float mat_sum(MAT *mat)
{
    float sum;
    for (size_t i = 0; i < mat->rows; i++)
    {
        for (size_t j = 0; j < mat->cols; j++)
        {
            sum += MAT_AT(*mat, i, j);
        }    
    }

    return sum;
}

void mat_scale(MAT *dst, MAT *mat, float value)
{
    for (size_t i=0; i < dst->rows; i++)
    {
        for (size_t j=0; j < dst->cols; j++)
        {
            MAT_AT(*dst, i, j) = value*MAT_AT(*mat, i, j);
        }
    }   
}

void mat_apply(MAT *dst, MAT *mat, float (*func)(float))
{
    for (size_t i=0; i < dst->rows; i++)
    {
        for (size_t j=0; j < dst->cols; j++)
        {
            MAT_AT(*dst, i, j) = func(MAT_AT(*mat, i, j));
        }
    }   
}

MAT mat_select(MAT *parent, size_t row_start, size_t row_count, size_t col_start, size_t col_count)
{
    MAT child = mat_alloc(col_count, row_count);
    for (size_t i = 0; i < row_count; i++)
    {
        for (size_t j = 0; j < col_count; j++)
        {
            MAT_AT(child, i, j) = MAT_AT(*parent, row_start + i, col_start + j);
        }
        
    }
    // mat_print(child);
    return child;
    
}

#endif //MAT_IMPLEMENTATION