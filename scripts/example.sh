python train.py --dataset cifar10 \
                --algorithm lp --cls_reg 0.0 \
                --epochs 100 --batch_size 200 --init_lr 0.005 --decay_step 5 --decay_rate 0.8 ;

python train.py --dataset cifar10 --cpt_path ./concept_bank/cifar10/cifar10_fine_selection_bar_3_num_20.pkl \
                --algorithm cbm --cls_reg 0.001 \
                --epochs 100 --batch_size 200 --init_lr 0.005 --decay_step 5 --decay_rate 0.8 ;

python train.py --dataset cifar10 --cpt_path ./concept_bank/cifar10/cifar10_rough_selection_bar_5.pkl \
                --algorithm mask --cls_reg 0.001 \
                --epochs 100 --batch_size 200 --init_lr 0.005 --decay_step 5 --decay_rate 0.8