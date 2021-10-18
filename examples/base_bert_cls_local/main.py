import sys

# train step
if "train" in sys.argv[1]:
    from lichee import train
    train.run()
# eval step
elif "eval" in sys.argv[1]:
    from lichee import eval
    eval.run()
# predict step
elif "predict" in sys.argv[1]:
    from lichee import predict
    predict.run()
