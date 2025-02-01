# Решение задачи линейного программирования методом Нестерова-Тодда

## Установка

0. Установите зависимости (пример для Arch Linux)

```
[...] $ sudo pacman -S blas superlu eigen
```

1. Клонируйте репозиторий

```
[/path/to/repo] $ git clone https://github.com/Semen-prog/lpsolver
```

2. Соберите проект

```
[...] $ mkdir build
[...] $ cd build
[.../build] $ cmake /path/to/repo/lpsolver
[.../make] $ make
```

3. Вы получите исполняемый файл lpsolver. Пользуйтесь!

## Использование

Программа получает на вход значения `n`, `m`, `A`, `b`, `c`, `x0`, `y0`, `s0` в следующем виде:

```
n m
A(1, 1) ... A(1, n)
...
A(m, 1) ... A(m, n)
b_1 ... b_m
c_1 ... c_n
x0_1 ... x0_n
y0_1 ... y0_m
s0_1 ... s0_n 
```