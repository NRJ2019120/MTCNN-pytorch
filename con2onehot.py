
def to_onehot(con_tensor):
    con = con_tensor.numpy()
    # print(con_tensor)
    print(con)
    print(con.size)
    batch = con.size
    for i in range(batch):
        if con[i][0] == 1:
            con[i] = [0,1]
        if con[i][0] == 0:
            print(con[i])
            con[i] = [1,0]
    return con



