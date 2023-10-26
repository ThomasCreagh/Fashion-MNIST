import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import json

loop = True

question = input("Do u want to train the model? (y/n): ")

if question == "y":
    epochs = int(input("How many passes do u want? (int): "))

    train_df = pd.read_csv('data/fashion-mnist_train.csv')
    test_df = pd.read_csv('data/fashion-mnist_test.csv')

    train_data = np.array(train_df)
    test_data = np.array(test_df)

    images, labels = train_data[:, 1:], train_data[:, 0]

    test = test_data[:, 1:]
    test = test.astype("float32") / 255

    images = images.astype("float32") / 255
    labels = np.eye(10)[labels]

    w_i_h = np.random.uniform(-0.5, 0.5, (20, 784))
    w_h_o = np.random.uniform(-0.5, 0.5, (10, 20))

    b_i_h = np.zeros((20, 1))
    b_h_o = np.zeros((10, 1))

    learn_rate = 0.01
    nr_correct = 0

    for epoch in range(epochs):
        for img, l in zip(images, labels):
            img.shape += (1,)
            l.shape += (1,)

            h_pre = b_i_h + w_i_h @ img
            h = 1 / (1 + np.exp(-h_pre))

            o_pre = b_h_o + w_h_o @ h
            o = 1 / (1 + np.exp(-o_pre))
            
            e = 1 / len(o) * np.sum((o - l) ** 2, axis=0)

            nr_correct += int(np.argmax(o) == np.argmax(l))

            delta_o = o - l
            w_h_o += -learn_rate * delta_o @ np.transpose(h)
            b_h_o += -learn_rate * delta_o
            delta_h = np.transpose(w_h_o) @ delta_o * (h * (1 - h))
            w_i_h += -learn_rate * delta_h @ np.transpose(img)
            b_i_h += -learn_rate * delta_h

        print(f"Epoch: {epoch}, Acc: {round((nr_correct / images.shape[0]) * 100, 2)}%")
        nr_correct = 0

    if input("Do u want to save this nural network? (y or n): ") == "y":
        # j_w_i_h = json.dumps()
        # j_w_h_o = json.dumps()
        # j_b_
        # with open("data/nn_saves.csv", "w") as r_f:
        #     r_f = json.writer(new_file)
        #     r_f.writerow(w_i_h)
        #     r_f.writerow(w_h_o)
        #     r_f.writerow(b_i_h)
        #     r_F.writerow(b_h_o)
        pass



elif question == "n":
    # with open("data/nn_saves.csv", "r") as csv_file:
    #     csv_reader = csv.reader(csv_file, delimiter='-')
    #     for rows in csv_reader:
    #         print(rows[1])
    #         # w_i_h = rows[0]
    #         # w_h_o = rows[1]
    #         # b_i_h = rows[2]
    #         # b_h_o = rows[3]
    pass
    
else:
    print("Please enter y or n")
    loop = False

# print(w_i_h)
# print(w_h_o)
# print(b_i_h)
# print(b_h_o)

if loop == True:
    while True:
        index = int(input("Enter a number (0 - 59999): "))
        img = test[index]
        plt.imshow(img.reshape(28, 28), cmap="Greys")

        h_pre = b_i_h + w_i_h @ img.reshape(784, 1)
        h = 1 / (1 + np.exp(-h_pre))
        o_pre = b_h_o + w_h_o @ h
        o = 1 / (1 + np.exp(-o_pre))
        if o.argmax() == 0: final_o = 'T-shirt'
        elif o.argmax() == 1: final_o = 'Trousers'
        elif o.argmax() == 2: final_o = 'Pullover'
        elif o.argmax() == 3: final_o = 'Dress'
        elif o.argmax() == 4: final_o = 'Coat'
        elif o.argmax() == 5: final_o = 'Sandals'
        elif o.argmax() == 6: final_o = 'Shirt'
        elif o.argmax() == 7: final_o = 'Sneakers'
        elif o.argmax() == 8: final_o = 'Bag'
        elif o.argmax() == 9: final_o = 'Ankle boots'

        plt.title(f"I think this is a {final_o}, am I right? :D")
        plt.show()
