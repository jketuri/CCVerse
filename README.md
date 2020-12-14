# CCVerse

You need Anaconda environment in Windows:

https://www.anaconda.com/products/individual

Run setup.cmd in Anaconda Prompt as Administrator.

In \*nix, run
setup.sh

Use Python 3.8 or newer.

This application uses as a base model an example from:

https://machinetalk.org/2019/02/08/text-generation-with-pytorch/

Run the application with:

CCVerse.cmd

or

CCVerse.sh

Default initial words are dream, light.
Other initial words can be given in command line e.g.:
CCVerse night sorrow

The application will download at the first time about 6gb word embeddings.
It will do this only once if directory .word_vectors_cache is not deleted.
Then the application will train 200 epochs and will output text at the same time
when loss is less than 0.7 and other conditions are met.
Output is also saved to the file output.txt.
