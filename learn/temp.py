class Cal:
    def __init__(self):
        self.num=0

    def add(self,x):
        self.num+=x
        return self.num

a=Cal().add(5)
print(f'a:{a}')
