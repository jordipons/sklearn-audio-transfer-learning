**Can I run it in collab?**

Yes, see a rudimentary example [here](https://colab.research.google.com/drive/194wo-G5fCZmqIFt0BDXZcpDpdlbU9Yns). If you are a colab ninja and want your example to appear here, please ping us!

**I get an error related to `tkinter` when installing the requirements. What shall I do?**

You need to install it in your operative system. In Ubuntu, do that: `sudo apt-get install python3-tk`.

**I cannot run this in my computer because the script consumes too much RAM memory. What shall I do?**

Reduce the batch size by properlly setting `batch_size`. The batch size defines the ammount of audios that are processed at once. The smaller it is, the lower the consumption of RAM memory.

When using the `openl3` feature extractor you will find that to solve your RAM issues you have to **increase** the batch size. We recommend to set it to 256. This is possibly due to a bug in the original Openl3 implementation.
