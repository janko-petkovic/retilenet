cd ..

python training.py MNIST BNLeNet True
python training.py FashionMNIST BNLeNet True
python training.py SVHN BNLeNet True

python training.py FashionMNIST LeNet True
python training.py SVHN LeNet True

python training.py MNIST Deep_RetiNet True
python training.py FashionMNIST Deep_RetiNet True
python training.py SVHN Deep_RetiNet True
