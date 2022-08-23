#!/usr/bin/python

from multiprocessing import Process


def fun(name, shit):
    print(f'hello {name} {shit}')

def main():

    p = Process(target=fun, args=('Peter', 'haha'))
    p.start()
    p.join()
    p = Process(target=fun, args=('Sarah', 'pig'))
    p.start()
    p.join()


if __name__ == '__main__':
    main()