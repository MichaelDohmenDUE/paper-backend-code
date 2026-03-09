


def fun(a=(), b=[]):
    a += (1,)
    b += [1]
    return a, b

fun()
print(fun())