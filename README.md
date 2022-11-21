# Telegram Notifier for Deep Learning
This is a simple script to send notifications to a Telegram group when a training process is finished. It is useful when you are training a model on a remote server and you want to be notified.

Example of usage:
```python
from telegram_notifier import bot, Store
from tqdm import tqdm


st = Store()
# values to send to the telegram bot
log_values = ["loss", "lm_loss"]
# values to save as model checkpoints
save_values = ["loss", "lm_loss"]

modes = ["train", "valid"]
...
for epoch in range(epochs):
    for mode in modes:
        # to deal with deque set maxlen
        maxlen = 2000 if mode == "train" else None
        st.reset(maxlen=maxlen)
        # add mode and epoch to the store
        st.add_value(mode), st.add_value(epoch)
        ...
        pbar = tqdm(enumerate(loader), total=len(loader))
        for idx, batch in pbar:
            # do something
            # ...
            # save the values
            outputs = model(inputs, labels=labels)
            loss = outputs[0]
            loss = st.add_value(loss)
            pbar.set_description(st.training_description)
        # save globals after each epoch
        st.save_global()
        # send a notification when the epoch is finished
        if mode == "valid":
            # get valid model name based on params
            name = st.get_output_string("filename", *save_values)
            # save weights / cfg / ... to a file
            model.save(name)
            message = st.get_output_string("message", *log_values)
            # send message
            bot.send_message(message)
            # send plots
            bot.send_plots(st.get_global(log_values))
```