# ArtML_text_generation
Simple implementation of the charbased LSTM used to generate text. Part of ML4Art / Art4ML class

# Installation:

- Set up the prerequisites: Python 3, Keras, Tensoflow, Numpy

  * install Python 3
  * `sudo apt-get install build-essential python-dev`
  * `sudo pip install tensorflow keras numpy`
 
# Use model:

  * train the model with `train_model_on_file(txt_file, name, epochs, batch_size)` (produces ".hdf5" files)
  * evaluate model with `use_model(txt_file, model_file)`
  * see codes in `wonderland.py` and `twinpeaks.py` for example usage
  
# Example outputs:

## Wonderland

 **model** = `wonderland-weights-41-1.2619-bigger.hdf5` (loss 1.2619)
 
 **seed** = `to show you! a little bright-eyed terrier, you know, with oh, such long curly brown hair! and it’ll be a large cat!`

 **generated text** = `‘i’m afraid i con’t know the sea,’ the ming raid to the jury, and alice was soon a pieht of the way the white rabbit, who was not a great heard a little walterss the whole parter of the gianter and the thoee gardeners, aut she was a little sather so coubers, and the thing said to herself, ‘i won’t the breatures out of this mouse- and what would be ouizes for to dome tp the matter would be all the caby, and she thought that it was anl reppled and she thought that it was all croken to doob that she was now a bot of the word in the way of expe, and was beginning to be a long way that she was now a bot of the word and was to the white rabbit, who was not a great heard a little was to the white rabbit, who was not a great heard a little was to the white rabbit, who was not a great heard a little was to the white rabbit, who was not a great heard a little was to the white rabbit, who was not a great heard a little was to the white rabbit, who was not a great heard a little`

_(it definitelly likes white rabbits... they appear in many generated fragments)_

## Twin Peaks

 **model** = `twinpeaks-weights-20-1.3289-bigger.hdf5` (loss 1.3289)
 
 **seed 1** = `listically could be.  laura comes to shelly in a panic. laura shelly, you really can hel`

 **generated text 1** =
  
   
         l me the tereyour hand ie a gook the breat.    
   
				   laura
			       (to herself)
			io it is in the bork sight.  there is
			she siog of the bare of the bar wour here
			and here and i was whth the breat.

				   laura
			       (to herself)
			io it is in the bork sight.  there is
			she siog of the bare of the bar wour here
			and here and i was whth the breat.

				   laura
			       (to herself)
			io it is in the bork sight.  there is
			she siog of the bare of the bar wour here
			and here and i was whth the breat.

				   laura
			       (to herself)
			io it is in the bork sight.  there is
			she siog of the bare of the bar wour here
			and here and i was whth the breat.

				   laura
			       (to herself)
			io it is in the bork sight.  there is
			she siog of the bare of the bar wour here
			and here and i was whth the breat.`

 **seed 2** = `wavy hair with a black full mustache to match, shiny silk shirt with silver strands sown in and and the pi`

 **generated text 2** = 
 
				   laura
			       (vm herself)
			the srart about the ban fornd and ie
			the riog of the fan and take to 			the ring of the fan and the way 			the doeat and the high ther wou
			the found about the srailer was 			the doeat.  i don't know wher is 			the ricture you ar a light of the 			srailer park.

				   laura
			       (vm herself)
			the srart about the ban fornd and ie
			the riog of the fan and take to 			the ring of the fan and the way 			the doeat and the high ther wou
			the found about the srailer was 			the doeat.  i don't know wher is 			the ricture you ar a light of the 			srailer park.

_(Seems to be mimicking at least the same structure as in the input file - of a theatrical play. But often the generated content is nonsence...)_

# Links:
Tutorial https://machinelearningmastery.com/text-generation-lstm-recurrent-neural-networks-python-keras/
