# Решение задачи линейного программирования методом Нестерова-Тодда

## Использование

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

## Генератор

Доступен генератор задач!