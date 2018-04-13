# KTH Deep Learning in Data Science - Assignment 4

The aim of this assignment is to train a RNN (recurrent neural network) to synthesize English text (from Harry Potter snippets) character by character and to synthesize Donald Trump Tweets. This was approached is two methods: with a vanilla RNN and a LSTM (long short-term memory).

Implemented in MATLAB from first principles. 

## Harry Potter Text

The following is a synthesized 1,000-character text made from the best vanila RNN model during training, based on the minimum loss value.  The model was trained for 30 epochs.

> . IWd thin to skill, rother, and cabes were diset of tomerhanering than theys if hims at the Deafly for hell, well oy him who poweo Bart9"
" yough huir of wink. . ?". gree was suwort -
"oTch poours, Harry, "I dobe right, how supports,reding tincinding. ."
"dice got in the pan what looked the Darks, "lear rouncond "ut which themiddly with the so llade.  They's  wongy?" sareed leacling the Derly windone when he did tooke out of toungry wotled to as Moasent. . ."
"It who ames and their.  in forlly.  They're - Xarned ie them dine was was fingerssoont to dope beather dir feet they wasn't have pand, eds hoaingnoor.  "allh very downorswerming a toward Yoors did to them it was sheed to sor thround into the so'he looked atw. "I Woast e told pass they houles to stair "jooks about all thiy thomer glans of your.". . . a sounkf of them. "yell was!" said slary ansider someone soing teeth turA!".  Over the Defully coursely. "It's going other now," said Mursy, as Beros told they reath Eated, be summore

The following is a synthesized 1,000-character text made from the best LSTM (*seq_length = 25, m = 512*) model during training, based on the minimum loss value.  

> Crouch's gargerer on his chpinged in the hand and friends belined silver thougmoh.  But her eyes worried a few look, looking earercsilf sobttacion know haidly always ramomely. nor'in, he began trueâingu" ; get Rot accoddly better Fred_get as he gavecher the carriage as
ungringingling at the RonYigg - they were the tome and reserfline, but theyole, and, befamee, he sack intonated for 7ursely the 3arklereringzaw 7as 6arken. or Vuros-- he seatel for them.
"Ho'lp haved you comestary to compentrated with the 1allowas up.
"Harry Potter madking pame," said Ron. "Hemento presubla lotioning in the solitwertsicully green /imbinatws and following, her and o persued Hogwarts until all as warnted abouiling at him, he rememberP for a lot hawnet, but in ther,' herefaterwalle liw his and Harryâcounting and all.  "and they- pranted not the perchal refinders undlious suddenly felled in the garVoernon, looking had colling at the Beauxbatons was haspily, and gold Bill grawnnd as they fursh over an e aro

## Donald Trump Tweets

The following are examples of synthesized tweets from the best LSTM model during training, based on the minimum loss value. The following hyperparameters were used: *seq_length = full tweet length, m=256*. Model was trained for 10 epochs.

> Thank you:
#MakeAmericaGreatAgain

> S\u2026 MAKE AMERICA GREAT AGAIN! we will receive the Obama Adm of mout people- than a great electing the tuck of your agai

> @drmistions joss. Tremendous officers. Please respect less out my Hillary Convisternine Tonight of @megynkelly so innested!                 

> @HillaryClinton is its way out of prime Hillary who have been asking the noth?

> @Rogand : Crooded Hillary is going to MAKE AMERICA GREAT AGAFS!
