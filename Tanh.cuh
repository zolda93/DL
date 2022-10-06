#ifndef _TANH_H
#define _TANH_H

#include "Layer.cuh"

namesapace DL
{
        class Tanh : public Layer
        {
                public:
                        Tanh(){}
                        ~Tanh(){}

                        virtual Tensor* forward(Tensor* x) override;
                        virtual Tensor* backward(Tensor* d) override;

                        virtual void init_forward(Tensor* x) override;
                        virtual void init_backward(Tensor* d) override;

        };

        void Tanh::init_forward(Tensor* x)
        {
                X = x;
                if(Z == nullptr)
                {
                        Z = new Tensor(X->shape());
                }else if(Z->batch() != X->batch())
                {
                        Z->reshape(X->shape());
                }
        }

        void Tanh::init_backward(Tensor* dz)
        {
                if(dX == nullptr)
                {
                        dX = new Tensor(X->shape());
                }else if(dX->batch() != X->batch())
                {
                        dX->reshape(X->batch());
                }
        }

        Tensor* Tanh::forward(Tensor* x)
        {
                return tanh_forward(Z,X);
        }

        Tensor* Tanh::backward(Tensor* dz)
        {
                return tanh_backward(dX,dz,Z);
        }
}
#endif
