scp -r ./* yme:~/thesis_autotune
ssh yme -t "
    cd thesis_autotune;
    docker build . -t nvidia-test;
    "

    #docker create -it --gpus all --name laplace2d nvidia-test bash;
    #docker cp laplace2d:/usr/src/laplace2d/results/laplace2d.png .
