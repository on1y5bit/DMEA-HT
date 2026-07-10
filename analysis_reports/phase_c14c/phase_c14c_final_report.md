# Phase C14-C Pairwise Ranking Inversion Decomposition

C14-C is analysis-only. No training, optimizer, backward pass, threshold tuning, label/split/task/manifest/report changes, or test-based selection occurred.

## Reproduction Gate

| seed | checkpoint_path | saved_prediction_rows | reproduced_prediction_rows | patient_id_match | label_match | max_abs_probability_difference | mean_abs_probability_difference | reproduction_pass | notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | runs/dmea_ht_v2_c13_temporal_focus_stress_seeds/checkpoints/seed_0_best.pt | 94 | 94 | 1 | 1 | 1.1102230246251565e-16 | 2.3400312420623313e-17 | 1 | eval + no_grad; character tokenizer; C13 checkpoint |
| 42 | runs/dmea_ht_v2_c13_temporal_focus_stress_seeds/checkpoints/seed_42_best.pt | 94 | 94 | 1 | 1 | 1.1102230246251565e-16 | 2.2588314197825658e-17 | 1 | eval + no_grad; character tokenizer; C13 checkpoint |
| 3407 | runs/dmea_ht_v2_c13_temporal_focus_stress_seeds/checkpoints/seed_3407_best.pt | 94 | 94 | 1 | 1 | 1.1102230246251565e-16 | 2.048450062057719e-17 | 1 | eval + no_grad; character tokenizer; C13 checkpoint |

Reproduction status: `PASS`.

## Contribution Reconstruction

| seed | patient_id | full_logit | classifier_equation_reconstructed_logit | classifier_equation_reconstruction_error | requested_additive_reconstruction_status | discordance_contribution | classifier_bias |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 10168610 | 0.15747468173503876 | 0.15747468918561935 | 7.450580596923828e-09 | unavailable_discordance_and_bias | unavailable | unavailable |
| 0 | 10012205 | -0.8558571934700012 | -0.8558571822941303 | 1.1175870895385742e-08 | unavailable_discordance_and_bias | unavailable | unavailable |
| 0 | 10138528 | -0.04569554328918457 | -0.04569550231099129 | 4.0978193283081055e-08 | unavailable_discordance_and_bias | unavailable | unavailable |
| 0 | 10022640 | 1.7438280582427979 | 1.7438280284404755 | -2.9802322387695312e-08 | unavailable_discordance_and_bias | unavailable | unavailable |
| 0 | 10007340 | -1.7043097019195557 | -1.704309731721878 | -2.9802322387695312e-08 | unavailable_discordance_and_bias | unavailable | unavailable |
| 0 | 10048335 | -0.463720440864563 | -0.46372049674391747 | -5.587935447692871e-08 | unavailable_discordance_and_bias | unavailable | unavailable |
| 0 | 10092491 | -2.335864782333374 | -2.335864670574665 | 1.1175870895385742e-07 | unavailable_discordance_and_bias | unavailable | unavailable |
| 0 | 10001110 | 0.15342950820922852 | 0.15342961251735687 | 1.043081283569336e-07 | unavailable_discordance_and_bias | unavailable | unavailable |
| 0 | 10151887 | -1.2415703535079956 | -1.2415703162550926 | 3.725290298461914e-08 | unavailable_discordance_and_bias | unavailable | unavailable |
| 0 | 10111615 | -1.464630126953125 | -1.4646301046013832 | 2.2351741790771484e-08 | unavailable_discordance_and_bias | unavailable | unavailable |
| 0 | 10091684 | 1.3160572052001953 | 1.3160571604967117 | -4.470348358154297e-08 | unavailable_discordance_and_bias | unavailable | unavailable |
| 0 | 10119735 | 4.121400833129883 | 4.121400818228722 | -1.4901161193847656e-08 | unavailable_discordance_and_bias | unavailable | unavailable |
| 0 | 10108395 | 0.09719538688659668 | 0.09719540178775787 | 1.4901161193847656e-08 | unavailable_discordance_and_bias | unavailable | unavailable |
| 0 | 10024132 | -1.1706385612487793 | -1.1706386990845203 | -1.3783574104309082e-07 | unavailable_discordance_and_bias | unavailable | unavailable |
| 0 | 10010767 | -1.4964842796325684 | -1.4964843541383743 | -7.450580596923828e-08 | unavailable_discordance_and_bias | unavailable | unavailable |
| 0 | 10089751 | -1.4891992807388306 | -1.4891992956399918 | -1.4901161193847656e-08 | unavailable_discordance_and_bias | unavailable | unavailable |
| 0 | 10030315 | -1.609613060951233 | -1.6096131205558777 | -5.960464477539063e-08 | unavailable_discordance_and_bias | unavailable | unavailable |
| 0 | 10099866 | -2.5938432216644287 | -2.5938432794064283 | -5.774199962615967e-08 | unavailable_discordance_and_bias | unavailable | unavailable |
| 0 | 10124993 | -0.3344968557357788 | -0.33449679613113403 | 5.960464477539063e-08 | unavailable_discordance_and_bias | unavailable | unavailable |
| 0 | 10145328 | 0.5861152410507202 | 0.5861151963472366 | -4.470348358154297e-08 | unavailable_discordance_and_bias | unavailable | unavailable |
| 0 | 10018899 | -2.28645920753479 | -2.2864591293036938 | 7.82310962677002e-08 | unavailable_discordance_and_bias | unavailable | unavailable |
| 0 | 10009149 | 0.7076072692871094 | 0.7076072543859482 | -1.4901161193847656e-08 | unavailable_discordance_and_bias | unavailable | unavailable |
| 0 | 10005340 | 0.23478853702545166 | 0.23478847742080688 | -5.960464477539063e-08 | unavailable_discordance_and_bias | unavailable | unavailable |
| 0 | 10068961 | -2.166071891784668 | -2.1660719215869904 | -2.9802322387695312e-08 | unavailable_discordance_and_bias | unavailable | unavailable |
| 0 | 10082765 | 0.04007863998413086 | 0.04007866978645325 | 2.9802322387695312e-08 | unavailable_discordance_and_bias | unavailable | unavailable |
| 0 | 10111232 | -1.767262578010559 | -1.7672626189887524 | -4.0978193283081055e-08 | unavailable_discordance_and_bias | unavailable | unavailable |
| 0 | 10127720 | -0.7711732983589172 | -0.7711732871830463 | 1.1175870895385742e-08 | unavailable_discordance_and_bias | unavailable | unavailable |
| 0 | 10103564 | -0.012377440929412842 | -0.012377455830574036 | -1.4901161193847656e-08 | unavailable_discordance_and_bias | unavailable | unavailable |
| 0 | 10034192 | -2.6507692337036133 | -2.6507692597806454 | -2.60770320892334e-08 | unavailable_discordance_and_bias | unavailable | unavailable |
| 0 | 10124320 | -2.1244750022888184 | -2.1244750767946243 | -7.450580596923828e-08 | unavailable_discordance_and_bias | unavailable | unavailable |
| 0 | 10064626 | 0.7093899250030518 | 0.7093900144100189 | 8.940696716308594e-08 | unavailable_discordance_and_bias | unavailable | unavailable |
| 0 | 10043013 | 0.6961038112640381 | 0.6961038261651993 | 1.4901161193847656e-08 | unavailable_discordance_and_bias | unavailable | unavailable |
| 0 | 10083859 | 0.44716179370880127 | 0.4471617490053177 | -4.470348358154297e-08 | unavailable_discordance_and_bias | unavailable | unavailable |
| 0 | 10163258 | 1.608910083770752 | 1.60891018435359 | 1.0058283805847168e-07 | unavailable_discordance_and_bias | unavailable | unavailable |
| 0 | 10068457 | -1.7162859439849854 | -1.7162858843803406 | 5.960464477539063e-08 | unavailable_discordance_and_bias | unavailable | unavailable |
| 0 | 10033121 | -2.266749382019043 | -2.266749296337366 | 8.568167686462402e-08 | unavailable_discordance_and_bias | unavailable | unavailable |
| 0 | 10135408 | 1.398343563079834 | 1.3983435779809952 | 1.4901161193847656e-08 | unavailable_discordance_and_bias | unavailable | unavailable |
| 0 | 10106168 | -0.4428281784057617 | -0.4428281933069229 | -1.4901161193847656e-08 | unavailable_discordance_and_bias | unavailable | unavailable |
| 0 | 10157441 | -0.8785498738288879 | -0.878549862653017 | 1.1175870895385742e-08 | unavailable_discordance_and_bias | unavailable | unavailable |
| 0 | 10027380 | -0.3543927073478699 | -0.354392696171999 | 1.1175870895385742e-08 | unavailable_discordance_and_bias | unavailable | unavailable |
| 0 | 10131359 | -0.2948122024536133 | -0.2948121167719364 | 8.568167686462402e-08 | unavailable_discordance_and_bias | unavailable | unavailable |
| 0 | 10034355 | -1.5644599199295044 | -1.5644598305225372 | 8.940696716308594e-08 | unavailable_discordance_and_bias | unavailable | unavailable |
| 0 | 10057459 | -0.43684113025665283 | -0.43684113025665283 | 0.0 | unavailable_discordance_and_bias | unavailable | unavailable |
| 0 | 10084329 | 0.6039555072784424 | 0.6039554327726364 | -7.450580596923828e-08 | unavailable_discordance_and_bias | unavailable | unavailable |
| 0 | 10016203 | -3.66353702545166 | -3.6635369807481766 | 4.470348358154297e-08 | unavailable_discordance_and_bias | unavailable | unavailable |
| 0 | 10144325 | -2.2547988891601562 | -2.2547989040613174 | -1.4901161193847656e-08 | unavailable_discordance_and_bias | unavailable | unavailable |
| 0 | 10164295 | 1.750809907913208 | 1.7508099526166916 | 4.470348358154297e-08 | unavailable_discordance_and_bias | unavailable | unavailable |
| 0 | 10048847 | -2.1821446418762207 | -2.1821444928646088 | 1.4901161193847656e-07 | unavailable_discordance_and_bias | unavailable | unavailable |
| 0 | 10093113 | 0.402296781539917 | 0.4022967666387558 | -1.4901161193847656e-08 | unavailable_discordance_and_bias | unavailable | unavailable |
| 0 | 10020343 | -0.35937297344207764 | -0.3593730218708515 | -4.842877388000488e-08 | unavailable_discordance_and_bias | unavailable | unavailable |
| 0 | 10063399 | 0.6877472400665283 | 0.6877472549676895 | 1.4901161193847656e-08 | unavailable_discordance_and_bias | unavailable | unavailable |
| 0 | 10015079 | -1.8190585374832153 | -1.8190585300326347 | 7.450580596923828e-09 | unavailable_discordance_and_bias | unavailable | unavailable |
| 0 | 10034913 | 0.7434886693954468 | 0.7434886656701565 | -3.725290298461914e-09 | unavailable_discordance_and_bias | unavailable | unavailable |
| 0 | 10130640 | -2.1815974712371826 | -2.181597501039505 | -2.9802322387695312e-08 | unavailable_discordance_and_bias | unavailable | unavailable |
| 0 | 10075536 | -1.1929569244384766 | -1.1929569244384766 | 0.0 | unavailable_discordance_and_bias | unavailable | unavailable |
| 0 | 10065841 | -0.9251287579536438 | -0.9251287803053856 | -2.2351741790771484e-08 | unavailable_discordance_and_bias | unavailable | unavailable |
| 0 | 10156499 | -0.09687209129333496 | -0.09687206521630287 | 2.60770320892334e-08 | unavailable_discordance_and_bias | unavailable | unavailable |
| 0 | 10038703 | -1.1252447366714478 | -1.125244677066803 | 5.960464477539063e-08 | unavailable_discordance_and_bias | unavailable | unavailable |
| 0 | 10003245 | -1.892168402671814 | -1.892168503254652 | -1.0058283805847168e-07 | unavailable_discordance_and_bias | unavailable | unavailable |
| 0 | 10066099 | -2.2157177925109863 | -2.2157179228961468 | -1.30385160446167e-07 | unavailable_discordance_and_bias | unavailable | unavailable |
| 0 | 10138067 | 0.46209609508514404 | 0.4620961658656597 | 7.078051567077637e-08 | unavailable_discordance_and_bias | unavailable | unavailable |
| 0 | 10023011 | 0.09009921550750732 | 0.09009917080402374 | -4.470348358154297e-08 | unavailable_discordance_and_bias | unavailable | unavailable |
| 0 | 10110001 | -3.0064828395843506 | -3.0064828358590603 | 3.725290298461914e-09 | unavailable_discordance_and_bias | unavailable | unavailable |
| 0 | 10007452 | -2.6308064460754395 | -2.6308066062629223 | -1.601874828338623e-07 | unavailable_discordance_and_bias | unavailable | unavailable |
| 0 | 10056854 | 1.1770657300949097 | 1.1770657487213612 | 1.862645149230957e-08 | unavailable_discordance_and_bias | unavailable | unavailable |
| 0 | 10013708 | -1.80633544921875 | -1.8063353896141052 | 5.960464477539063e-08 | unavailable_discordance_and_bias | unavailable | unavailable |
| 0 | 10038097 | 0.3333929777145386 | 0.3333929777145386 | 0.0 | unavailable_discordance_and_bias | unavailable | unavailable |
| 0 | 10025710 | -1.4220161437988281 | -1.422016218304634 | -7.450580596923828e-08 | unavailable_discordance_and_bias | unavailable | unavailable |
| 0 | 10064351 | -0.2863194942474365 | -0.28631946071982384 | 3.3527612686157227e-08 | unavailable_discordance_and_bias | unavailable | unavailable |
| 0 | 10031361 | -1.8535488843917847 | -1.8535488843917847 | 0.0 | unavailable_discordance_and_bias | unavailable | unavailable |
| 0 | 10132330 | 0.6522623896598816 | 0.6522623896598816 | 0.0 | unavailable_discordance_and_bias | unavailable | unavailable |
| 0 | 10042173 | -1.091034173965454 | -1.0910342819988728 | -1.0803341865539551e-07 | unavailable_discordance_and_bias | unavailable | unavailable |
| 0 | 10084278 | -0.2953658103942871 | -0.2953658849000931 | -7.450580596923828e-08 | unavailable_discordance_and_bias | unavailable | unavailable |
| 0 | 10019805 | -1.264675259590149 | -1.2646752707660198 | -1.1175870895385742e-08 | unavailable_discordance_and_bias | unavailable | unavailable |
| 0 | 10079757 | 1.0870039463043213 | 1.087003916501999 | -2.9802322387695312e-08 | unavailable_discordance_and_bias | unavailable | unavailable |
| 0 | 10085505 | -2.554635763168335 | -2.554635778069496 | -1.4901161193847656e-08 | unavailable_discordance_and_bias | unavailable | unavailable |
| 0 | 10045442 | 0.44122469425201416 | 0.4412246346473694 | -5.960464477539063e-08 | unavailable_discordance_and_bias | unavailable | unavailable |
| 0 | 10004992 | -1.6047804355621338 | -1.6047804616391659 | -2.60770320892334e-08 | unavailable_discordance_and_bias | unavailable | unavailable |
| 0 | 10151451 | -0.2326112985610962 | -0.2326113060116768 | -7.450580596923828e-09 | unavailable_discordance_and_bias | unavailable | unavailable |
| 0 | 10007169 | -1.7134885787963867 | -1.7134885527193546 | 2.60770320892334e-08 | unavailable_discordance_and_bias | unavailable | unavailable |
| 0 | 10117252 | -0.8289389610290527 | -0.8289389312267303 | 2.9802322387695312e-08 | unavailable_discordance_and_bias | unavailable | unavailable |
| 0 | 10066912 | 0.707756519317627 | 0.7077565006911755 | -1.862645149230957e-08 | unavailable_discordance_and_bias | unavailable | unavailable |
| 0 | 10032546 | -1.0746724605560303 | -1.0746724642813206 | -3.725290298461914e-09 | unavailable_discordance_and_bias | unavailable | unavailable |
| 0 | 10052254 | -2.2631828784942627 | -2.2631829380989075 | -5.960464477539063e-08 | unavailable_discordance_and_bias | unavailable | unavailable |
| 0 | 10067220 | -0.045722365379333496 | -0.04572242125868797 | -5.587935447692871e-08 | unavailable_discordance_and_bias | unavailable | unavailable |
| 0 | 10051496 | -1.4784436225891113 | -1.4784437119960785 | -8.940696716308594e-08 | unavailable_discordance_and_bias | unavailable | unavailable |
| 0 | 10064537 | -0.39555859565734863 | -0.39555859938263893 | -3.725290298461914e-09 | unavailable_discordance_and_bias | unavailable | unavailable |
| 0 | 10005075 | -0.4597259759902954 | -0.4597260095179081 | -3.3527612686157227e-08 | unavailable_discordance_and_bias | unavailable | unavailable |
| 0 | 10098069 | 0.059729933738708496 | 0.059729982167482376 | 4.842877388000488e-08 | unavailable_discordance_and_bias | unavailable | unavailable |
| 0 | 10137578 | 0.1183476448059082 | 0.11834767088294029 | 2.60770320892334e-08 | unavailable_discordance_and_bias | unavailable | unavailable |
| 0 | 10132146 | -0.5233085751533508 | -0.5233085714280605 | 3.725290298461914e-09 | unavailable_discordance_and_bias | unavailable | unavailable |
| 0 | 10074227 | -1.555281639099121 | -1.5552817918360233 | -1.5273690223693848e-07 | unavailable_discordance_and_bias | unavailable | unavailable |
| 0 | 10014141 | -1.018141269683838 | -1.0181413479149342 | -7.82310962677002e-08 | unavailable_discordance_and_bias | unavailable | unavailable |
| 0 | 10083858 | -1.023444414138794 | -1.0234445221722126 | -1.0803341865539551e-07 | unavailable_discordance_and_bias | unavailable | unavailable |
| 42 | 10168610 | -0.08602261543273926 | -0.08602255582809448 | 5.960464477539063e-08 | unavailable_discordance_and_bias | unavailable | unavailable |
| 42 | 10012205 | 0.6463332176208496 | 0.6463331952691078 | -2.2351741790771484e-08 | unavailable_discordance_and_bias | unavailable | unavailable |
| 42 | 10138528 | 1.1325335502624512 | 1.1325336024165154 | 5.21540641784668e-08 | unavailable_discordance_and_bias | unavailable | unavailable |
| 42 | 10022640 | 3.657935380935669 | 3.6579353883862495 | 7.450580596923828e-09 | unavailable_discordance_and_bias | unavailable | unavailable |
| 42 | 10007340 | -0.17295566201210022 | -0.17295563966035843 | 2.2351741790771484e-08 | unavailable_discordance_and_bias | unavailable | unavailable |
| 42 | 10048335 | -0.7729120850563049 | -0.7729120720177889 | 1.30385160446167e-08 | unavailable_discordance_and_bias | unavailable | unavailable |
| 42 | 10092491 | -2.8189101219177246 | -2.8189100474119186 | 7.450580596923828e-08 | unavailable_discordance_and_bias | unavailable | unavailable |
| 42 | 10001110 | 1.5866553783416748 | 1.5866553708910942 | -7.450580596923828e-09 | unavailable_discordance_and_bias | unavailable | unavailable |
| 42 | 10151887 | -0.37681418657302856 | -0.37681417539715767 | 1.1175870895385742e-08 | unavailable_discordance_and_bias | unavailable | unavailable |
| 42 | 10111615 | -0.44024309515953064 | -0.44024310261011124 | -7.450580596923828e-09 | unavailable_discordance_and_bias | unavailable | unavailable |
| 42 | 10091684 | 2.7440590858459473 | 2.744059167802334 | 8.195638656616211e-08 | unavailable_discordance_and_bias | unavailable | unavailable |
| 42 | 10119735 | 5.856990814208984 | 5.8569908663630486 | 5.21540641784668e-08 | unavailable_discordance_and_bias | unavailable | unavailable |
| 42 | 10108395 | 1.614013671875 | 1.6140136700123549 | -1.862645149230957e-09 | unavailable_discordance_and_bias | unavailable | unavailable |
| 42 | 10024132 | 0.18438246846199036 | 0.18438245356082916 | -1.4901161193847656e-08 | unavailable_discordance_and_bias | unavailable | unavailable |
| 42 | 10010767 | -2.598432779312134 | -2.5984326973557472 | 8.195638656616211e-08 | unavailable_discordance_and_bias | unavailable | unavailable |
| 42 | 10089751 | -0.8262658715248108 | -0.8262658752501011 | -3.725290298461914e-09 | unavailable_discordance_and_bias | unavailable | unavailable |
| 42 | 10030315 | -2.0070509910583496 | -2.007050946354866 | 4.470348358154297e-08 | unavailable_discordance_and_bias | unavailable | unavailable |
| 42 | 10099866 | -3.4921467304229736 | -3.4921466894447803 | 4.0978193283081055e-08 | unavailable_discordance_and_bias | unavailable | unavailable |
| 42 | 10124993 | 1.012037992477417 | 1.0120379459112883 | -4.6566128730773926e-08 | unavailable_discordance_and_bias | unavailable | unavailable |
| 42 | 10145328 | 2.3862810134887695 | 2.386281132698059 | 1.1920928955078125e-07 | unavailable_discordance_and_bias | unavailable | unavailable |
| 42 | 10018899 | -0.8752287030220032 | -0.8752286955714226 | 7.450580596923828e-09 | unavailable_discordance_and_bias | unavailable | unavailable |
| 42 | 10009149 | 2.628814697265625 | 2.6288147047162056 | 7.450580596923828e-09 | unavailable_discordance_and_bias | unavailable | unavailable |
| 42 | 10005340 | 1.2343345880508423 | 1.2343345656991005 | -2.2351741790771484e-08 | unavailable_discordance_and_bias | unavailable | unavailable |
| 42 | 10068961 | -2.13946533203125 | -2.1394654400646687 | -1.0803341865539551e-07 | unavailable_discordance_and_bias | unavailable | unavailable |
| 42 | 10082765 | 0.8851519823074341 | 0.8851519599556923 | -2.2351741790771484e-08 | unavailable_discordance_and_bias | unavailable | unavailable |
| 42 | 10111232 | -1.393723487854004 | -1.3937235474586487 | -5.960464477539063e-08 | unavailable_discordance_and_bias | unavailable | unavailable |
| 42 | 10127720 | 0.2025931179523468 | 0.2025931105017662 | -7.450580596923828e-09 | unavailable_discordance_and_bias | unavailable | unavailable |
| 42 | 10103564 | 1.3168426752090454 | 1.3168426603078842 | -1.4901161193847656e-08 | unavailable_discordance_and_bias | unavailable | unavailable |
| 42 | 10034192 | -2.570122241973877 | -2.5701223835349083 | -1.4156103134155273e-07 | unavailable_discordance_and_bias | unavailable | unavailable |
| 42 | 10124320 | -2.4214320182800293 | -2.421431904658675 | 1.1362135410308838e-07 | unavailable_discordance_and_bias | unavailable | unavailable |
| 42 | 10064626 | 1.612121820449829 | 1.612121805548668 | -1.4901161193847656e-08 | unavailable_discordance_and_bias | unavailable | unavailable |
| 42 | 10043013 | 1.5142614841461182 | 1.5142614617943764 | -2.2351741790771484e-08 | unavailable_discordance_and_bias | unavailable | unavailable |
| 42 | 10083859 | 2.003464698791504 | 2.003464810550213 | 1.1175870895385742e-07 | unavailable_discordance_and_bias | unavailable | unavailable |
| 42 | 10163258 | 3.2301974296569824 | 3.230197452008724 | 2.2351741790771484e-08 | unavailable_discordance_and_bias | unavailable | unavailable |
| 42 | 10068457 | -0.4371802806854248 | -0.4371802732348442 | 7.450580596923828e-09 | unavailable_discordance_and_bias | unavailable | unavailable |
| 42 | 10033121 | -1.728386640548706 | -1.7283864989876747 | 1.4156103134155273e-07 | unavailable_discordance_and_bias | unavailable | unavailable |
| 42 | 10135408 | 2.3572816848754883 | 2.3572817519307137 | 6.705522537231445e-08 | unavailable_discordance_and_bias | unavailable | unavailable |
| 42 | 10106168 | 1.3568840026855469 | 1.356883980333805 | -2.2351741790771484e-08 | unavailable_discordance_and_bias | unavailable | unavailable |
| 42 | 10157441 | 0.3911796808242798 | 0.3911796808242798 | 0.0 | unavailable_discordance_and_bias | unavailable | unavailable |
| 42 | 10027380 | 0.40900158882141113 | 0.40900157392024994 | -1.4901161193847656e-08 | unavailable_discordance_and_bias | unavailable | unavailable |
| 42 | 10131359 | 0.07143011689186096 | 0.07143011689186096 | 0.0 | unavailable_discordance_and_bias | unavailable | unavailable |
| 42 | 10034355 | -1.4831886291503906 | -1.4831886067986488 | 2.2351741790771484e-08 | unavailable_discordance_and_bias | unavailable | unavailable |
| 42 | 10057459 | 1.6802997589111328 | 1.680299699306488 | -5.960464477539063e-08 | unavailable_discordance_and_bias | unavailable | unavailable |
| 42 | 10084329 | 2.1509933471679688 | 2.1509932577610016 | -8.940696716308594e-08 | unavailable_discordance_and_bias | unavailable | unavailable |
| 42 | 10016203 | -3.7883193492889404 | -3.7883192971348763 | 5.21540641784668e-08 | unavailable_discordance_and_bias | unavailable | unavailable |
| 42 | 10144325 | -3.128793716430664 | -3.128793776035309 | -5.960464477539063e-08 | unavailable_discordance_and_bias | unavailable | unavailable |
| 42 | 10164295 | 3.584787607192993 | 3.584787666797638 | 5.960464477539063e-08 | unavailable_discordance_and_bias | unavailable | unavailable |
| 42 | 10048847 | -2.5976967811584473 | -2.5976969078183174 | -1.2665987014770508e-07 | unavailable_discordance_and_bias | unavailable | unavailable |
| 42 | 10093113 | 1.4890141487121582 | 1.4890141859650612 | 3.725290298461914e-08 | unavailable_discordance_and_bias | unavailable | unavailable |
| 42 | 10020343 | -0.19160506129264832 | -0.1916050761938095 | -1.4901161193847656e-08 | unavailable_discordance_and_bias | unavailable | unavailable |
| 42 | 10063399 | 2.0320520401000977 | 2.032051958143711 | -8.195638656616211e-08 | unavailable_discordance_and_bias | unavailable | unavailable |
| 42 | 10015079 | -1.8465532064437866 | -1.8465532511472702 | -4.470348358154297e-08 | unavailable_discordance_and_bias | unavailable | unavailable |
| 42 | 10034913 | 1.9592809677124023 | 1.9592809453606606 | -2.2351741790771484e-08 | unavailable_discordance_and_bias | unavailable | unavailable |
| 42 | 10130640 | -1.6375726461410522 | -1.6375726610422134 | -1.4901161193847656e-08 | unavailable_discordance_and_bias | unavailable | unavailable |
| 42 | 10075536 | -0.691075325012207 | -0.6910753548145294 | -2.9802322387695312e-08 | unavailable_discordance_and_bias | unavailable | unavailable |
| 42 | 10065841 | -0.8307664394378662 | -0.8307664338499308 | 5.587935447692871e-09 | unavailable_discordance_and_bias | unavailable | unavailable |
| 42 | 10156499 | 1.2331504821777344 | 1.233150452375412 | -2.9802322387695312e-08 | unavailable_discordance_and_bias | unavailable | unavailable |
| 42 | 10038703 | -0.88170325756073 | -0.8817032501101494 | 7.450580596923828e-09 | unavailable_discordance_and_bias | unavailable | unavailable |
| 42 | 10003245 | -1.823843002319336 | -1.8238429799675941 | 2.2351741790771484e-08 | unavailable_discordance_and_bias | unavailable | unavailable |
| 42 | 10066099 | -2.0207135677337646 | -2.020713619887829 | -5.21540641784668e-08 | unavailable_discordance_and_bias | unavailable | unavailable |
| 42 | 10138067 | 1.0493886470794678 | 1.0493885949254036 | -5.21540641784668e-08 | unavailable_discordance_and_bias | unavailable | unavailable |
| 42 | 10023011 | 0.858289897441864 | 0.8582899160683155 | 1.862645149230957e-08 | unavailable_discordance_and_bias | unavailable | unavailable |
| 42 | 10110001 | -2.633906126022339 | -2.633906163275242 | -3.725290298461914e-08 | unavailable_discordance_and_bias | unavailable | unavailable |
| 42 | 10007452 | -2.3240599632263184 | -2.32405998557806 | -2.2351741790771484e-08 | unavailable_discordance_and_bias | unavailable | unavailable |
| 42 | 10056854 | 2.328432083129883 | 2.328431986272335 | -9.685754776000977e-08 | unavailable_discordance_and_bias | unavailable | unavailable |
| 42 | 10013708 | -0.885954737663269 | -0.8859547656029463 | -2.7939677238464355e-08 | unavailable_discordance_and_bias | unavailable | unavailable |
| 42 | 10038097 | 1.540209174156189 | 1.5402091965079308 | 2.2351741790771484e-08 | unavailable_discordance_and_bias | unavailable | unavailable |
| 42 | 10025710 | -0.626796305179596 | -0.6267963200807571 | -1.4901161193847656e-08 | unavailable_discordance_and_bias | unavailable | unavailable |
| 42 | 10064351 | 0.5463250875473022 | 0.5463250949978828 | 7.450580596923828e-09 | unavailable_discordance_and_bias | unavailable | unavailable |
| 42 | 10031361 | -1.8912291526794434 | -1.8912291247397661 | 2.7939677238464355e-08 | unavailable_discordance_and_bias | unavailable | unavailable |
| 42 | 10132330 | 1.3141942024230957 | 1.3141941726207733 | -2.9802322387695312e-08 | unavailable_discordance_and_bias | unavailable | unavailable |
| 42 | 10042173 | -0.30699077248573303 | -0.3069907873868942 | -1.4901161193847656e-08 | unavailable_discordance_and_bias | unavailable | unavailable |
| 42 | 10084278 | 0.08051794767379761 | 0.08051794767379761 | 0.0 | unavailable_discordance_and_bias | unavailable | unavailable |
| 42 | 10019805 | -0.3578365445137024 | -0.3578365594148636 | -1.4901161193847656e-08 | unavailable_discordance_and_bias | unavailable | unavailable |
| 42 | 10079757 | 2.531102180480957 | 2.53110221773386 | 3.725290298461914e-08 | unavailable_discordance_and_bias | unavailable | unavailable |
| 42 | 10085505 | -2.292790651321411 | -2.2927905842661858 | 6.705522537231445e-08 | unavailable_discordance_and_bias | unavailable | unavailable |
| 42 | 10045442 | 1.4920367002487183 | 1.4920367300510406 | 2.9802322387695312e-08 | unavailable_discordance_and_bias | unavailable | unavailable |
| 42 | 10004992 | -2.222055435180664 | -2.222055569291115 | -1.341104507446289e-07 | unavailable_discordance_and_bias | unavailable | unavailable |
| 42 | 10151451 | 0.5356696248054504 | 0.5356696154922247 | -9.313225746154785e-09 | unavailable_discordance_and_bias | unavailable | unavailable |
| 42 | 10007169 | -2.319495916366577 | -2.3194959983229637 | -8.195638656616211e-08 | unavailable_discordance_and_bias | unavailable | unavailable |
| 42 | 10117252 | -0.30205801129341125 | -0.30205801129341125 | 0.0 | unavailable_discordance_and_bias | unavailable | unavailable |
| 42 | 10066912 | 2.0681312084198 | 2.068131186068058 | -2.2351741790771484e-08 | unavailable_discordance_and_bias | unavailable | unavailable |
| 42 | 10032546 | -0.5818053483963013 | -0.5818053726106882 | -2.421438694000244e-08 | unavailable_discordance_and_bias | unavailable | unavailable |
| 42 | 10052254 | -2.428663730621338 | -2.428663820028305 | -8.940696716308594e-08 | unavailable_discordance_and_bias | unavailable | unavailable |
| 42 | 10067220 | 0.8348398208618164 | 0.8348398506641388 | 2.9802322387695312e-08 | unavailable_discordance_and_bias | unavailable | unavailable |
| 42 | 10051496 | -0.8810884952545166 | -0.8810885325074196 | -3.725290298461914e-08 | unavailable_discordance_and_bias | unavailable | unavailable |
| 42 | 10064537 | 0.09114532917737961 | 0.09114532917737961 | 0.0 | unavailable_discordance_and_bias | unavailable | unavailable |
| 42 | 10005075 | 0.49137061834335327 | 0.49137061834335327 | 0.0 | unavailable_discordance_and_bias | unavailable | unavailable |
| 42 | 10098069 | 1.0795761346817017 | 1.0795760843902826 | -5.029141902923584e-08 | unavailable_discordance_and_bias | unavailable | unavailable |
| 42 | 10137578 | 1.1207995414733887 | 1.1207995191216469 | -2.2351741790771484e-08 | unavailable_discordance_and_bias | unavailable | unavailable |
| 42 | 10132146 | -0.03711302578449249 | -0.03711302392184734 | 1.862645149230957e-09 | unavailable_discordance_and_bias | unavailable | unavailable |
| 42 | 10074227 | -1.0313926935195923 | -1.031392678618431 | 1.4901161193847656e-08 | unavailable_discordance_and_bias | unavailable | unavailable |
| 42 | 10014141 | 0.2356835901737213 | 0.2356835938990116 | 3.725290298461914e-09 | unavailable_discordance_and_bias | unavailable | unavailable |
| 42 | 10083858 | -0.15040013194084167 | -0.15040014311671257 | -1.1175870895385742e-08 | unavailable_discordance_and_bias | unavailable | unavailable |
| 3407 | 10168610 | -0.6209453344345093 | -0.6209453605115414 | -2.60770320892334e-08 | unavailable_discordance_and_bias | unavailable | unavailable |
| 3407 | 10012205 | -0.19403566420078278 | -0.19403565488755703 | 9.313225746154785e-09 | unavailable_discordance_and_bias | unavailable | unavailable |
| 3407 | 10138528 | -0.39747798442840576 | -0.39747798442840576 | 0.0 | unavailable_discordance_and_bias | unavailable | unavailable |
| 3407 | 10022640 | 2.71236515045166 | 2.7123651653528214 | 1.4901161193847656e-08 | unavailable_discordance_and_bias | unavailable | unavailable |
| 3407 | 10007340 | -0.9010568261146545 | -0.9010568112134933 | 1.4901161193847656e-08 | unavailable_discordance_and_bias | unavailable | unavailable |
| 3407 | 10048335 | -1.2205746173858643 | -1.2205746416002512 | -2.421438694000244e-08 | unavailable_discordance_and_bias | unavailable | unavailable |
| 3407 | 10092491 | -2.535339593887329 | -2.5353396236896515 | -2.9802322387695312e-08 | unavailable_discordance_and_bias | unavailable | unavailable |
| 3407 | 10001110 | 0.76273512840271 | 0.76273512840271 | 0.0 | unavailable_discordance_and_bias | unavailable | unavailable |
| 3407 | 10151887 | -1.17633056640625 | -1.1763305515050888 | 1.4901161193847656e-08 | unavailable_discordance_and_bias | unavailable | unavailable |
| 3407 | 10111615 | -1.6972239017486572 | -1.6972239390015602 | -3.725290298461914e-08 | unavailable_discordance_and_bias | unavailable | unavailable |
| 3407 | 10091684 | 1.9129022359848022 | 1.9129022061824799 | -2.9802322387695312e-08 | unavailable_discordance_and_bias | unavailable | unavailable |
| 3407 | 10119735 | 4.256524085998535 | 4.256523832678795 | -2.5331974029541016e-07 | unavailable_discordance_and_bias | unavailable | unavailable |
| 3407 | 10108395 | 0.8283742666244507 | 0.828374233096838 | -3.3527612686157227e-08 | unavailable_discordance_and_bias | unavailable | unavailable |
| 3407 | 10024132 | -0.8986684083938599 | -0.8986684307456017 | -2.2351741790771484e-08 | unavailable_discordance_and_bias | unavailable | unavailable |
| 3407 | 10010767 | -2.4716880321502686 | -2.471688073128462 | -4.0978193283081055e-08 | unavailable_discordance_and_bias | unavailable | unavailable |
| 3407 | 10089751 | -1.0804628133773804 | -1.080462772399187 | 4.0978193283081055e-08 | unavailable_discordance_and_bias | unavailable | unavailable |
| 3407 | 10030315 | -1.869469165802002 | -1.869469165802002 | 0.0 | unavailable_discordance_and_bias | unavailable | unavailable |
| 3407 | 10099866 | -2.8727264404296875 | -2.8727265000343323 | -5.960464477539063e-08 | unavailable_discordance_and_bias | unavailable | unavailable |
| 3407 | 10124993 | 0.43231719732284546 | 0.43231725320219994 | 5.587935447692871e-08 | unavailable_discordance_and_bias | unavailable | unavailable |
| 3407 | 10145328 | 1.1330552101135254 | 1.133055169135332 | -4.0978193283081055e-08 | unavailable_discordance_and_bias | unavailable | unavailable |
| 3407 | 10018899 | -1.5768368244171143 | -1.5768368542194366 | -2.9802322387695312e-08 | unavailable_discordance_and_bias | unavailable | unavailable |
| 3407 | 10009149 | 1.6768609285354614 | 1.6768609695136547 | 4.0978193283081055e-08 | unavailable_discordance_and_bias | unavailable | unavailable |
| 3407 | 10005340 | 0.7829768657684326 | 0.7829768396914005 | -2.60770320892334e-08 | unavailable_discordance_and_bias | unavailable | unavailable |
| 3407 | 10068961 | -2.344728708267212 | -2.3447288125753403 | -1.043081283569336e-07 | unavailable_discordance_and_bias | unavailable | unavailable |
| 3407 | 10082765 | 0.1709570288658142 | 0.1709570288658142 | 0.0 | unavailable_discordance_and_bias | unavailable | unavailable |
| 3407 | 10111232 | -1.9424443244934082 | -1.942444309592247 | 1.4901161193847656e-08 | unavailable_discordance_and_bias | unavailable | unavailable |
| 3407 | 10127720 | -0.6770846843719482 | -0.6770847328007221 | -4.842877388000488e-08 | unavailable_discordance_and_bias | unavailable | unavailable |
| 3407 | 10103564 | 0.2461973875761032 | 0.2461974136531353 | 2.60770320892334e-08 | unavailable_discordance_and_bias | unavailable | unavailable |
| 3407 | 10034192 | -3.0167808532714844 | -3.016780987381935 | -1.341104507446289e-07 | unavailable_discordance_and_bias | unavailable | unavailable |
| 3407 | 10124320 | -2.4695823192596436 | -2.469582311809063 | 7.450580596923828e-09 | unavailable_discordance_and_bias | unavailable | unavailable |
| 3407 | 10064626 | 0.9111380577087402 | 0.9111380279064178 | -2.9802322387695312e-08 | unavailable_discordance_and_bias | unavailable | unavailable |
| 3407 | 10043013 | 0.9781268239021301 | 0.9781268239021301 | 0.0 | unavailable_discordance_and_bias | unavailable | unavailable |
| 3407 | 10083859 | 1.1428184509277344 | 1.1428184807300568 | 2.9802322387695312e-08 | unavailable_discordance_and_bias | unavailable | unavailable |
| 3407 | 10163258 | 2.087097406387329 | 2.0870974212884903 | 1.4901161193847656e-08 | unavailable_discordance_and_bias | unavailable | unavailable |
| 3407 | 10068457 | -1.132704257965088 | -1.1327042542397976 | 3.725290298461914e-09 | unavailable_discordance_and_bias | unavailable | unavailable |
| 3407 | 10033121 | -3.050154447555542 | -3.050154536962509 | -8.940696716308594e-08 | unavailable_discordance_and_bias | unavailable | unavailable |
| 3407 | 10135408 | 1.8056613206863403 | 1.8056612759828568 | -4.470348358154297e-08 | unavailable_discordance_and_bias | unavailable | unavailable |
| 3407 | 10106168 | 0.14365486800670624 | 0.14365487918257713 | 1.1175870895385742e-08 | unavailable_discordance_and_bias | unavailable | unavailable |
| 3407 | 10157441 | -0.09996472299098969 | -0.09996471926569939 | 3.725290298461914e-09 | unavailable_discordance_and_bias | unavailable | unavailable |
| 3407 | 10027380 | 0.0492396354675293 | 0.0492396354675293 | 0.0 | unavailable_discordance_and_bias | unavailable | unavailable |
| 3407 | 10131359 | -0.007145218551158905 | -0.007145218551158905 | 0.0 | unavailable_discordance_and_bias | unavailable | unavailable |
| 3407 | 10034355 | -1.541651964187622 | -1.5416520088911057 | -4.470348358154297e-08 | unavailable_discordance_and_bias | unavailable | unavailable |
| 3407 | 10057459 | 0.3518362045288086 | 0.35183618031442165 | -2.421438694000244e-08 | unavailable_discordance_and_bias | unavailable | unavailable |
| 3407 | 10084329 | 1.5020121335983276 | 1.5020121484994888 | 1.4901161193847656e-08 | unavailable_discordance_and_bias | unavailable | unavailable |
| 3407 | 10016203 | -3.138633966445923 | -3.138633891940117 | 7.450580596923828e-08 | unavailable_discordance_and_bias | unavailable | unavailable |
| 3407 | 10144325 | -2.4365782737731934 | -2.436578180640936 | 9.313225746154785e-08 | unavailable_discordance_and_bias | unavailable | unavailable |
| 3407 | 10164295 | 2.0598719120025635 | 2.059871941804886 | 2.9802322387695312e-08 | unavailable_discordance_and_bias | unavailable | unavailable |
| 3407 | 10048847 | -2.0070881843566895 | -2.0070882439613342 | -5.960464477539063e-08 | unavailable_discordance_and_bias | unavailable | unavailable |
| 3407 | 10093113 | 0.38941094279289246 | 0.38941095024347305 | 7.450580596923828e-09 | unavailable_discordance_and_bias | unavailable | unavailable |
| 3407 | 10020343 | -0.026176825165748596 | -0.026176825165748596 | 0.0 | unavailable_discordance_and_bias | unavailable | unavailable |
| 3407 | 10063399 | 0.7919422388076782 | 0.7919422425329685 | 3.725290298461914e-09 | unavailable_discordance_and_bias | unavailable | unavailable |
| 3407 | 10015079 | -1.924739956855774 | -1.9247400164604187 | -5.960464477539063e-08 | unavailable_discordance_and_bias | unavailable | unavailable |
| 3407 | 10034913 | 1.0490243434906006 | 1.0490242652595043 | -7.82310962677002e-08 | unavailable_discordance_and_bias | unavailable | unavailable |
| 3407 | 10130640 | -2.153096914291382 | -2.1530968993902206 | 1.4901161193847656e-08 | unavailable_discordance_and_bias | unavailable | unavailable |
| 3407 | 10075536 | -1.12896728515625 | -1.12896728515625 | 0.0 | unavailable_discordance_and_bias | unavailable | unavailable |
| 3407 | 10065841 | -0.93461674451828 | -0.93461674451828 | 0.0 | unavailable_discordance_and_bias | unavailable | unavailable |
| 3407 | 10156499 | 0.28967714309692383 | 0.28967714309692383 | 0.0 | unavailable_discordance_and_bias | unavailable | unavailable |
| 3407 | 10038703 | -1.0885533094406128 | -1.088553275913 | 3.3527612686157227e-08 | unavailable_discordance_and_bias | unavailable | unavailable |
| 3407 | 10003245 | -1.9891222715377808 | -1.9891222808510065 | -9.313225746154785e-09 | unavailable_discordance_and_bias | unavailable | unavailable |
| 3407 | 10066099 | -2.6234383583068848 | -2.6234382688999176 | 8.940696716308594e-08 | unavailable_discordance_and_bias | unavailable | unavailable |
| 3407 | 10138067 | 0.4365343153476715 | 0.4365343153476715 | 0.0 | unavailable_discordance_and_bias | unavailable | unavailable |
| 3407 | 10023011 | -0.28817081451416016 | -0.28817084431648254 | -2.9802322387695312e-08 | unavailable_discordance_and_bias | unavailable | unavailable |
| 3407 | 10110001 | -3.459507942199707 | -3.459508180618286 | -2.384185791015625e-07 | unavailable_discordance_and_bias | unavailable | unavailable |
| 3407 | 10007452 | -3.0258424282073975 | -3.025842532515526 | -1.043081283569336e-07 | unavailable_discordance_and_bias | unavailable | unavailable |
| 3407 | 10056854 | 1.1075491905212402 | 1.107549175620079 | -1.4901161193847656e-08 | unavailable_discordance_and_bias | unavailable | unavailable |
| 3407 | 10013708 | -1.3562793731689453 | -1.3562794029712677 | -2.9802322387695312e-08 | unavailable_discordance_and_bias | unavailable | unavailable |
| 3407 | 10038097 | 0.9997206926345825 | 0.9997206926345825 | 0.0 | unavailable_discordance_and_bias | unavailable | unavailable |
| 3407 | 10025710 | -0.7323907613754272 | -0.732390746474266 | 1.4901161193847656e-08 | unavailable_discordance_and_bias | unavailable | unavailable |
| 3407 | 10064351 | -0.22967574000358582 | -0.22967574000358582 | 0.0 | unavailable_discordance_and_bias | unavailable | unavailable |
| 3407 | 10031361 | -2.012791633605957 | -2.0127916671335697 | -3.3527612686157227e-08 | unavailable_discordance_and_bias | unavailable | unavailable |
| 3407 | 10132330 | 0.3764180541038513 | 0.3764180690050125 | 1.4901161193847656e-08 | unavailable_discordance_and_bias | unavailable | unavailable |
| 3407 | 10042173 | -0.552773118019104 | -0.5527731254696846 | -7.450580596923828e-09 | unavailable_discordance_and_bias | unavailable | unavailable |
| 3407 | 10084278 | 0.1887141764163971 | 0.1887141801416874 | 3.725290298461914e-09 | unavailable_discordance_and_bias | unavailable | unavailable |
| 3407 | 10019805 | -1.2135313749313354 | -1.213531345129013 | 2.9802322387695312e-08 | unavailable_discordance_and_bias | unavailable | unavailable |
| 3407 | 10079757 | 1.6949468851089478 | 1.694946900010109 | 1.4901161193847656e-08 | unavailable_discordance_and_bias | unavailable | unavailable |
| 3407 | 10085505 | -2.9062488079071045 | -2.906248927116394 | -1.1920928955078125e-07 | unavailable_discordance_and_bias | unavailable | unavailable |
| 3407 | 10045442 | 0.7673354148864746 | 0.767335407435894 | -7.450580596923828e-09 | unavailable_discordance_and_bias | unavailable | unavailable |
| 3407 | 10004992 | -2.3881640434265137 | -2.3881641067564487 | -6.332993507385254e-08 | unavailable_discordance_and_bias | unavailable | unavailable |
| 3407 | 10151451 | -0.23993059992790222 | -0.23993060737848282 | -7.450580596923828e-09 | unavailable_discordance_and_bias | unavailable | unavailable |
| 3407 | 10007169 | -2.8713479042053223 | -2.871347874403 | 2.9802322387695312e-08 | unavailable_discordance_and_bias | unavailable | unavailable |
| 3407 | 10117252 | -0.9092357158660889 | -0.9092357233166695 | -7.450580596923828e-09 | unavailable_discordance_and_bias | unavailable | unavailable |
| 3407 | 10066912 | 1.2769242525100708 | 1.2769242599606514 | 7.450580596923828e-09 | unavailable_discordance_and_bias | unavailable | unavailable |
| 3407 | 10032546 | -0.5366096496582031 | -0.5366096161305904 | 3.3527612686157227e-08 | unavailable_discordance_and_bias | unavailable | unavailable |
| 3407 | 10052254 | -2.3316080570220947 | -2.331608023494482 | 3.3527612686157227e-08 | unavailable_discordance_and_bias | unavailable | unavailable |
| 3407 | 10067220 | -0.055422715842723846 | -0.05542270839214325 | 7.450580596923828e-09 | unavailable_discordance_and_bias | unavailable | unavailable |
| 3407 | 10051496 | -1.7455635070800781 | -1.745563443750143 | 6.332993507385254e-08 | unavailable_discordance_and_bias | unavailable | unavailable |
| 3407 | 10064537 | -0.03625768423080444 | -0.03625766560435295 | 1.862645149230957e-08 | unavailable_discordance_and_bias | unavailable | unavailable |
| 3407 | 10005075 | -0.036025211215019226 | -0.036025214940309525 | -3.725290298461914e-09 | unavailable_discordance_and_bias | unavailable | unavailable |
| 3407 | 10098069 | 0.05440078675746918 | 0.05440077185630798 | -1.4901161193847656e-08 | unavailable_discordance_and_bias | unavailable | unavailable |
| 3407 | 10137578 | 0.0234978049993515 | 0.0234978087246418 | 3.725290298461914e-09 | unavailable_discordance_and_bias | unavailable | unavailable |
| 3407 | 10132146 | -0.06352407485246658 | -0.06352406740188599 | 7.450580596923828e-09 | unavailable_discordance_and_bias | unavailable | unavailable |
| 3407 | 10074227 | -1.7877931594848633 | -1.7877930868417025 | 7.264316082000732e-08 | unavailable_discordance_and_bias | unavailable | unavailable |
| 3407 | 10014141 | -0.36758002638816833 | -0.36758001148700714 | 1.4901161193847656e-08 | unavailable_discordance_and_bias | unavailable | unavailable |
| 3407 | 10083858 | -0.47442734241485596 | -0.47442733868956566 | 3.725290298461914e-09 | unavailable_discordance_and_bias | unavailable | unavailable |

The current classifier equation reconstructs the final logit using image, text, bio, anchor/synergy, and negative-evidence terms with zero numerical error. Requested strict additive attribution remains unavailable because the model does not expose a separable discordance contribution and classifier bias.

## Pairwise Inversions

- Expected pairs per seed: `2209`; observed pair rows: `6627`.
- Inversion rows: `885`.
| positive_patient_id | negative_patient_id | inversion_count | seed_count | image_opposed_count | image_repair_count | text_driven_count | fusion_interaction_count | inversion_group |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 10003245 | 10001110 | 3 | 3 | 0 | 0 | 3 | 0 | all_seed_inversion |
| 10003245 | 10004992 | 1 | 3 | 1 | 1 | 0 | 0 | single_seed_inversion |
| 10003245 | 10005075 | 3 | 3 | 0 | 0 | 3 | 0 | all_seed_inversion |
| 10003245 | 10007169 | 1 | 3 | 0 | 0 | 1 | 0 | single_seed_inversion |
| 10003245 | 10007340 | 3 | 3 | 0 | 0 | 3 | 0 | all_seed_inversion |
| 10003245 | 10007452 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10003245 | 10009149 | 3 | 3 | 0 | 0 | 3 | 0 | all_seed_inversion |
| 10003245 | 10010767 | 1 | 3 | 1 | 1 | 0 | 0 | single_seed_inversion |
| 10003245 | 10013708 | 3 | 3 | 3 | 0 | 0 | 0 | all_seed_inversion |
| 10003245 | 10015079 | 2 | 3 | 2 | 2 | 0 | 0 | majority_seed_inversion |
| 10003245 | 10016203 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10003245 | 10018899 | 2 | 3 | 2 | 0 | 0 | 0 | majority_seed_inversion |
| 10003245 | 10019805 | 3 | 3 | 1 | 0 | 2 | 0 | all_seed_inversion |
| 10003245 | 10020343 | 3 | 3 | 0 | 0 | 3 | 0 | all_seed_inversion |
| 10003245 | 10023011 | 3 | 3 | 3 | 0 | 0 | 0 | all_seed_inversion |
| 10003245 | 10025710 | 3 | 3 | 2 | 0 | 1 | 0 | all_seed_inversion |
| 10003245 | 10027380 | 3 | 3 | 0 | 0 | 3 | 0 | all_seed_inversion |
| 10003245 | 10030315 | 2 | 3 | 2 | 2 | 0 | 0 | majority_seed_inversion |
| 10003245 | 10031361 | 1 | 3 | 1 | 1 | 0 | 0 | single_seed_inversion |
| 10003245 | 10032546 | 3 | 3 | 1 | 0 | 2 | 0 | all_seed_inversion |
| 10003245 | 10033121 | 1 | 3 | 1 | 1 | 0 | 0 | single_seed_inversion |
| 10003245 | 10034192 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10003245 | 10034355 | 3 | 3 | 3 | 3 | 0 | 0 | all_seed_inversion |
| 10003245 | 10038703 | 3 | 3 | 0 | 0 | 3 | 0 | all_seed_inversion |
| 10003245 | 10043013 | 3 | 3 | 0 | 0 | 3 | 0 | all_seed_inversion |
| 10003245 | 10048335 | 3 | 3 | 1 | 0 | 2 | 0 | all_seed_inversion |
| 10003245 | 10048847 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10003245 | 10051496 | 3 | 3 | 3 | 1 | 0 | 0 | all_seed_inversion |
| 10003245 | 10052254 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10003245 | 10065841 | 3 | 3 | 3 | 0 | 0 | 0 | all_seed_inversion |
| 10003245 | 10066099 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10003245 | 10068457 | 3 | 3 | 3 | 0 | 0 | 0 | all_seed_inversion |
| 10003245 | 10068961 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10003245 | 10074227 | 3 | 3 | 2 | 1 | 1 | 0 | all_seed_inversion |
| 10003245 | 10083858 | 3 | 3 | 0 | 0 | 3 | 0 | all_seed_inversion |
| 10003245 | 10084278 | 3 | 3 | 1 | 0 | 2 | 0 | all_seed_inversion |
| 10003245 | 10085505 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10003245 | 10089751 | 3 | 3 | 3 | 0 | 0 | 0 | all_seed_inversion |
| 10003245 | 10092491 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10003245 | 10099866 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10003245 | 10106168 | 3 | 3 | 0 | 0 | 3 | 0 | all_seed_inversion |
| 10003245 | 10111232 | 3 | 3 | 2 | 2 | 1 | 0 | all_seed_inversion |
| 10003245 | 10111615 | 3 | 3 | 2 | 1 | 1 | 0 | all_seed_inversion |
| 10003245 | 10124320 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10003245 | 10130640 | 1 | 3 | 1 | 1 | 0 | 0 | single_seed_inversion |
| 10003245 | 10137578 | 3 | 3 | 0 | 0 | 3 | 0 | all_seed_inversion |
| 10003245 | 10144325 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10005340 | 10001110 | 1 | 3 | 0 | 0 | 1 | 0 | single_seed_inversion |
| 10005340 | 10004992 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10005340 | 10005075 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10005340 | 10007169 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10005340 | 10007340 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10005340 | 10007452 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10005340 | 10009149 | 3 | 3 | 0 | 0 | 3 | 0 | all_seed_inversion |
| 10005340 | 10010767 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10005340 | 10013708 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10005340 | 10015079 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10005340 | 10016203 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10005340 | 10018899 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10005340 | 10019805 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10005340 | 10020343 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10005340 | 10023011 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10005340 | 10025710 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10005340 | 10027380 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10005340 | 10030315 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10005340 | 10031361 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10005340 | 10032546 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10005340 | 10033121 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10005340 | 10034192 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10005340 | 10034355 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10005340 | 10038703 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10005340 | 10043013 | 3 | 3 | 0 | 0 | 3 | 0 | all_seed_inversion |
| 10005340 | 10048335 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10005340 | 10048847 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10005340 | 10051496 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10005340 | 10052254 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10005340 | 10065841 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10005340 | 10066099 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10005340 | 10068457 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10005340 | 10068961 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10005340 | 10074227 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10005340 | 10083858 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10005340 | 10084278 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10005340 | 10085505 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10005340 | 10089751 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10005340 | 10092491 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10005340 | 10099866 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10005340 | 10106168 | 1 | 3 | 0 | 0 | 1 | 0 | single_seed_inversion |
| 10005340 | 10111232 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10005340 | 10111615 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10005340 | 10124320 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10005340 | 10130640 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10005340 | 10137578 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10005340 | 10144325 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10012205 | 10001110 | 3 | 3 | 0 | 0 | 3 | 0 | all_seed_inversion |
| 10012205 | 10004992 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10012205 | 10005075 | 2 | 3 | 0 | 0 | 2 | 0 | majority_seed_inversion |
| 10012205 | 10007169 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10012205 | 10007340 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10012205 | 10007452 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10012205 | 10009149 | 3 | 3 | 0 | 0 | 3 | 0 | all_seed_inversion |
| 10012205 | 10010767 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10012205 | 10013708 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10012205 | 10015079 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10012205 | 10016203 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10012205 | 10018899 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10012205 | 10019805 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10012205 | 10020343 | 2 | 3 | 0 | 0 | 1 | 1 | majority_seed_inversion |
| 10012205 | 10023011 | 2 | 3 | 1 | 0 | 0 | 1 | majority_seed_inversion |
| 10012205 | 10025710 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10012205 | 10027380 | 2 | 3 | 0 | 0 | 2 | 0 | majority_seed_inversion |
| 10012205 | 10030315 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10012205 | 10031361 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10012205 | 10032546 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10012205 | 10033121 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10012205 | 10034192 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10012205 | 10034355 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10012205 | 10038703 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10012205 | 10043013 | 3 | 3 | 0 | 0 | 3 | 0 | all_seed_inversion |
| 10012205 | 10048335 | 1 | 3 | 0 | 0 | 1 | 0 | single_seed_inversion |
| 10012205 | 10048847 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10012205 | 10051496 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10012205 | 10052254 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10012205 | 10065841 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10012205 | 10066099 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10012205 | 10068457 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10012205 | 10068961 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10012205 | 10074227 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10012205 | 10083858 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10012205 | 10084278 | 2 | 3 | 1 | 0 | 1 | 0 | majority_seed_inversion |
| 10012205 | 10085505 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10012205 | 10089751 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10012205 | 10092491 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10012205 | 10099866 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10012205 | 10106168 | 3 | 3 | 0 | 0 | 3 | 0 | all_seed_inversion |
| 10012205 | 10111232 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10012205 | 10111615 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10012205 | 10124320 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10012205 | 10130640 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10012205 | 10137578 | 3 | 3 | 1 | 0 | 2 | 0 | all_seed_inversion |
| 10012205 | 10144325 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10014141 | 10001110 | 3 | 3 | 0 | 0 | 3 | 0 | all_seed_inversion |
| 10014141 | 10004992 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10014141 | 10005075 | 3 | 3 | 2 | 0 | 1 | 0 | all_seed_inversion |
| 10014141 | 10007169 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10014141 | 10007340 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10014141 | 10007452 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10014141 | 10009149 | 3 | 3 | 0 | 0 | 3 | 0 | all_seed_inversion |
| 10014141 | 10010767 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10014141 | 10013708 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10014141 | 10015079 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10014141 | 10016203 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10014141 | 10018899 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10014141 | 10019805 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10014141 | 10020343 | 2 | 3 | 1 | 0 | 1 | 0 | majority_seed_inversion |
| 10014141 | 10023011 | 3 | 3 | 3 | 1 | 0 | 0 | all_seed_inversion |
| 10014141 | 10025710 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10014141 | 10027380 | 3 | 3 | 1 | 1 | 2 | 0 | all_seed_inversion |
| 10014141 | 10030315 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10014141 | 10031361 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10014141 | 10032546 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10014141 | 10033121 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10014141 | 10034192 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10014141 | 10034355 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10014141 | 10038703 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10014141 | 10043013 | 3 | 3 | 0 | 0 | 3 | 0 | all_seed_inversion |
| 10014141 | 10048335 | 1 | 3 | 1 | 1 | 0 | 0 | single_seed_inversion |
| 10014141 | 10048847 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10014141 | 10051496 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10014141 | 10052254 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10014141 | 10065841 | 1 | 3 | 1 | 1 | 0 | 0 | single_seed_inversion |
| 10014141 | 10066099 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10014141 | 10068457 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10014141 | 10068961 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10014141 | 10074227 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10014141 | 10083858 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10014141 | 10084278 | 2 | 3 | 2 | 0 | 0 | 0 | majority_seed_inversion |
| 10014141 | 10085505 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10014141 | 10089751 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10014141 | 10092491 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10014141 | 10099866 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10014141 | 10106168 | 3 | 3 | 2 | 2 | 1 | 0 | all_seed_inversion |
| 10014141 | 10111232 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10014141 | 10111615 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10014141 | 10124320 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10014141 | 10130640 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10014141 | 10137578 | 3 | 3 | 2 | 2 | 1 | 0 | all_seed_inversion |
| 10014141 | 10144325 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10022640 | 10001110 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10022640 | 10004992 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10022640 | 10005075 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10022640 | 10007169 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10022640 | 10007340 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10022640 | 10007452 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10022640 | 10009149 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10022640 | 10010767 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10022640 | 10013708 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10022640 | 10015079 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10022640 | 10016203 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10022640 | 10018899 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10022640 | 10019805 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10022640 | 10020343 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10022640 | 10023011 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10022640 | 10025710 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10022640 | 10027380 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10022640 | 10030315 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10022640 | 10031361 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10022640 | 10032546 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10022640 | 10033121 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10022640 | 10034192 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10022640 | 10034355 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10022640 | 10038703 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10022640 | 10043013 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10022640 | 10048335 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10022640 | 10048847 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10022640 | 10051496 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10022640 | 10052254 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10022640 | 10065841 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10022640 | 10066099 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10022640 | 10068457 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10022640 | 10068961 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10022640 | 10074227 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10022640 | 10083858 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10022640 | 10084278 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10022640 | 10085505 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10022640 | 10089751 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10022640 | 10092491 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10022640 | 10099866 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10022640 | 10106168 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10022640 | 10111232 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10022640 | 10111615 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10022640 | 10124320 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10022640 | 10130640 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10022640 | 10137578 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10022640 | 10144325 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10024132 | 10001110 | 3 | 3 | 0 | 0 | 3 | 0 | all_seed_inversion |
| 10024132 | 10004992 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10024132 | 10005075 | 3 | 3 | 1 | 0 | 2 | 0 | all_seed_inversion |
| 10024132 | 10007169 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10024132 | 10007340 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10024132 | 10007452 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10024132 | 10009149 | 3 | 3 | 0 | 0 | 3 | 0 | all_seed_inversion |
| 10024132 | 10010767 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10024132 | 10013708 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10024132 | 10015079 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10024132 | 10016203 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10024132 | 10018899 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10024132 | 10019805 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10024132 | 10020343 | 2 | 3 | 1 | 0 | 1 | 0 | majority_seed_inversion |
| 10024132 | 10023011 | 3 | 3 | 2 | 0 | 0 | 1 | all_seed_inversion |
| 10024132 | 10025710 | 1 | 3 | 0 | 0 | 1 | 0 | single_seed_inversion |
| 10024132 | 10027380 | 3 | 3 | 1 | 0 | 2 | 0 | all_seed_inversion |
| 10024132 | 10030315 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10024132 | 10031361 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10024132 | 10032546 | 2 | 3 | 0 | 0 | 2 | 0 | majority_seed_inversion |
| 10024132 | 10033121 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10024132 | 10034192 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10024132 | 10034355 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10024132 | 10038703 | 1 | 3 | 0 | 0 | 1 | 0 | single_seed_inversion |
| 10024132 | 10043013 | 3 | 3 | 0 | 0 | 3 | 0 | all_seed_inversion |
| 10024132 | 10048335 | 1 | 3 | 0 | 0 | 1 | 0 | single_seed_inversion |
| 10024132 | 10048847 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10024132 | 10051496 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10024132 | 10052254 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10024132 | 10065841 | 1 | 3 | 1 | 0 | 0 | 0 | single_seed_inversion |
| 10024132 | 10066099 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10024132 | 10068457 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10024132 | 10068961 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10024132 | 10074227 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10024132 | 10083858 | 2 | 3 | 0 | 0 | 2 | 0 | majority_seed_inversion |
| 10024132 | 10084278 | 2 | 3 | 1 | 0 | 1 | 0 | majority_seed_inversion |
| 10024132 | 10085505 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10024132 | 10089751 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10024132 | 10092491 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10024132 | 10099866 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10024132 | 10106168 | 3 | 3 | 0 | 0 | 3 | 0 | all_seed_inversion |
| 10024132 | 10111232 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10024132 | 10111615 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10024132 | 10124320 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10024132 | 10130640 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10024132 | 10137578 | 3 | 3 | 1 | 0 | 2 | 0 | all_seed_inversion |
| 10024132 | 10144325 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10034913 | 10001110 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10034913 | 10004992 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10034913 | 10005075 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10034913 | 10007169 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10034913 | 10007340 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10034913 | 10007452 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10034913 | 10009149 | 2 | 3 | 0 | 0 | 2 | 0 | majority_seed_inversion |
| 10034913 | 10010767 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10034913 | 10013708 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10034913 | 10015079 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10034913 | 10016203 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10034913 | 10018899 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10034913 | 10019805 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10034913 | 10020343 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10034913 | 10023011 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10034913 | 10025710 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10034913 | 10027380 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10034913 | 10030315 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10034913 | 10031361 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10034913 | 10032546 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10034913 | 10033121 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10034913 | 10034192 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10034913 | 10034355 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10034913 | 10038703 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10034913 | 10043013 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10034913 | 10048335 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10034913 | 10048847 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10034913 | 10051496 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10034913 | 10052254 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10034913 | 10065841 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10034913 | 10066099 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10034913 | 10068457 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10034913 | 10068961 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10034913 | 10074227 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10034913 | 10083858 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10034913 | 10084278 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10034913 | 10085505 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10034913 | 10089751 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10034913 | 10092491 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10034913 | 10099866 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10034913 | 10106168 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10034913 | 10111232 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10034913 | 10111615 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10034913 | 10124320 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10034913 | 10130640 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10034913 | 10137578 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10034913 | 10144325 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10038097 | 10001110 | 1 | 3 | 0 | 0 | 1 | 0 | single_seed_inversion |
| 10038097 | 10004992 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10038097 | 10005075 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10038097 | 10007169 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10038097 | 10007340 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10038097 | 10007452 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10038097 | 10009149 | 3 | 3 | 0 | 0 | 3 | 0 | all_seed_inversion |
| 10038097 | 10010767 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10038097 | 10013708 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10038097 | 10015079 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10038097 | 10016203 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10038097 | 10018899 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10038097 | 10019805 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10038097 | 10020343 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10038097 | 10023011 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10038097 | 10025710 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10038097 | 10027380 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10038097 | 10030315 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10038097 | 10031361 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10038097 | 10032546 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10038097 | 10033121 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10038097 | 10034192 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10038097 | 10034355 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10038097 | 10038703 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10038097 | 10043013 | 1 | 3 | 0 | 0 | 1 | 0 | single_seed_inversion |
| 10038097 | 10048335 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10038097 | 10048847 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10038097 | 10051496 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10038097 | 10052254 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10038097 | 10065841 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10038097 | 10066099 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10038097 | 10068457 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10038097 | 10068961 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10038097 | 10074227 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10038097 | 10083858 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10038097 | 10084278 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10038097 | 10085505 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10038097 | 10089751 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10038097 | 10092491 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10038097 | 10099866 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10038097 | 10106168 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10038097 | 10111232 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10038097 | 10111615 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10038097 | 10124320 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10038097 | 10130640 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10038097 | 10137578 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10038097 | 10144325 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10042173 | 10001110 | 3 | 3 | 0 | 0 | 3 | 0 | all_seed_inversion |
| 10042173 | 10004992 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10042173 | 10005075 | 3 | 3 | 1 | 0 | 2 | 0 | all_seed_inversion |
| 10042173 | 10007169 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10042173 | 10007340 | 1 | 3 | 0 | 0 | 1 | 0 | single_seed_inversion |
| 10042173 | 10007452 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10042173 | 10009149 | 3 | 3 | 0 | 0 | 3 | 0 | all_seed_inversion |
| 10042173 | 10010767 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10042173 | 10013708 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10042173 | 10015079 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10042173 | 10016203 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10042173 | 10018899 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10042173 | 10019805 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10042173 | 10020343 | 3 | 3 | 1 | 0 | 2 | 0 | all_seed_inversion |
| 10042173 | 10023011 | 3 | 3 | 3 | 1 | 0 | 0 | all_seed_inversion |
| 10042173 | 10025710 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10042173 | 10027380 | 3 | 3 | 0 | 0 | 3 | 0 | all_seed_inversion |
| 10042173 | 10030315 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10042173 | 10031361 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10042173 | 10032546 | 2 | 3 | 1 | 1 | 1 | 0 | majority_seed_inversion |
| 10042173 | 10033121 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10042173 | 10034192 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10042173 | 10034355 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10042173 | 10038703 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10042173 | 10043013 | 3 | 3 | 0 | 0 | 3 | 0 | all_seed_inversion |
| 10042173 | 10048335 | 1 | 3 | 0 | 0 | 1 | 0 | single_seed_inversion |
| 10042173 | 10048847 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10042173 | 10051496 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10042173 | 10052254 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10042173 | 10065841 | 1 | 3 | 1 | 1 | 0 | 0 | single_seed_inversion |
| 10042173 | 10066099 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10042173 | 10068457 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10042173 | 10068961 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10042173 | 10074227 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10042173 | 10083858 | 3 | 3 | 0 | 0 | 2 | 1 | all_seed_inversion |
| 10042173 | 10084278 | 3 | 3 | 1 | 0 | 2 | 0 | all_seed_inversion |
| 10042173 | 10085505 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10042173 | 10089751 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10042173 | 10092491 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10042173 | 10099866 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10042173 | 10106168 | 3 | 3 | 1 | 1 | 2 | 0 | all_seed_inversion |
| 10042173 | 10111232 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10042173 | 10111615 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10042173 | 10124320 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10042173 | 10130640 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10042173 | 10137578 | 3 | 3 | 1 | 0 | 2 | 0 | all_seed_inversion |
| 10042173 | 10144325 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10045442 | 10001110 | 1 | 3 | 0 | 0 | 1 | 0 | single_seed_inversion |
| 10045442 | 10004992 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10045442 | 10005075 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10045442 | 10007169 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10045442 | 10007340 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10045442 | 10007452 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10045442 | 10009149 | 3 | 3 | 0 | 0 | 3 | 0 | all_seed_inversion |
| 10045442 | 10010767 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10045442 | 10013708 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10045442 | 10015079 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10045442 | 10016203 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10045442 | 10018899 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10045442 | 10019805 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10045442 | 10020343 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10045442 | 10023011 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10045442 | 10025710 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10045442 | 10027380 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10045442 | 10030315 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10045442 | 10031361 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10045442 | 10032546 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10045442 | 10033121 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10045442 | 10034192 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10045442 | 10034355 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10045442 | 10038703 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10045442 | 10043013 | 3 | 3 | 0 | 0 | 3 | 0 | all_seed_inversion |
| 10045442 | 10048335 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10045442 | 10048847 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10045442 | 10051496 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10045442 | 10052254 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10045442 | 10065841 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10045442 | 10066099 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10045442 | 10068457 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10045442 | 10068961 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10045442 | 10074227 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10045442 | 10083858 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10045442 | 10084278 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10045442 | 10085505 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10045442 | 10089751 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10045442 | 10092491 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10045442 | 10099866 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10045442 | 10106168 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10045442 | 10111232 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10045442 | 10111615 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10045442 | 10124320 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10045442 | 10130640 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10045442 | 10137578 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10045442 | 10144325 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10056854 | 10001110 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10056854 | 10004992 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10056854 | 10005075 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10056854 | 10007169 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10056854 | 10007340 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10056854 | 10007452 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10056854 | 10009149 | 2 | 3 | 0 | 0 | 2 | 0 | majority_seed_inversion |
| 10056854 | 10010767 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10056854 | 10013708 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10056854 | 10015079 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10056854 | 10016203 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10056854 | 10018899 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10056854 | 10019805 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10056854 | 10020343 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10056854 | 10023011 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10056854 | 10025710 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10056854 | 10027380 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10056854 | 10030315 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10056854 | 10031361 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10056854 | 10032546 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10056854 | 10033121 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10056854 | 10034192 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10056854 | 10034355 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10056854 | 10038703 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10056854 | 10043013 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10056854 | 10048335 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10056854 | 10048847 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10056854 | 10051496 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10056854 | 10052254 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10056854 | 10065841 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10056854 | 10066099 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10056854 | 10068457 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10056854 | 10068961 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10056854 | 10074227 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10056854 | 10083858 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10056854 | 10084278 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10056854 | 10085505 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10056854 | 10089751 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10056854 | 10092491 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10056854 | 10099866 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10056854 | 10106168 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10056854 | 10111232 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10056854 | 10111615 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10056854 | 10124320 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10056854 | 10130640 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10056854 | 10137578 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10056854 | 10144325 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10057459 | 10001110 | 2 | 3 | 0 | 0 | 2 | 0 | majority_seed_inversion |
| 10057459 | 10004992 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10057459 | 10005075 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10057459 | 10007169 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10057459 | 10007340 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10057459 | 10007452 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10057459 | 10009149 | 3 | 3 | 0 | 0 | 3 | 0 | all_seed_inversion |
| 10057459 | 10010767 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10057459 | 10013708 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10057459 | 10015079 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10057459 | 10016203 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10057459 | 10018899 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10057459 | 10019805 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10057459 | 10020343 | 1 | 3 | 0 | 0 | 0 | 1 | single_seed_inversion |
| 10057459 | 10023011 | 1 | 3 | 0 | 0 | 0 | 1 | single_seed_inversion |
| 10057459 | 10025710 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10057459 | 10027380 | 1 | 3 | 0 | 0 | 1 | 0 | single_seed_inversion |
| 10057459 | 10030315 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10057459 | 10031361 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10057459 | 10032546 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10057459 | 10033121 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10057459 | 10034192 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10057459 | 10034355 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10057459 | 10038703 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10057459 | 10043013 | 2 | 3 | 0 | 0 | 2 | 0 | majority_seed_inversion |
| 10057459 | 10048335 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10057459 | 10048847 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10057459 | 10051496 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10057459 | 10052254 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10057459 | 10065841 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10057459 | 10066099 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10057459 | 10068457 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10057459 | 10068961 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10057459 | 10074227 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10057459 | 10083858 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10057459 | 10084278 | 1 | 3 | 1 | 0 | 0 | 0 | single_seed_inversion |
| 10057459 | 10085505 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10057459 | 10089751 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10057459 | 10092491 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10057459 | 10099866 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10057459 | 10106168 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10057459 | 10111232 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10057459 | 10111615 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10057459 | 10124320 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10057459 | 10130640 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10057459 | 10137578 | 1 | 3 | 0 | 0 | 1 | 0 | single_seed_inversion |
| 10057459 | 10144325 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10063399 | 10001110 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10063399 | 10004992 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10063399 | 10005075 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10063399 | 10007169 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10063399 | 10007340 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10063399 | 10007452 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10063399 | 10009149 | 3 | 3 | 0 | 0 | 3 | 0 | all_seed_inversion |
| 10063399 | 10010767 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10063399 | 10013708 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10063399 | 10015079 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10063399 | 10016203 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10063399 | 10018899 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10063399 | 10019805 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10063399 | 10020343 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10063399 | 10023011 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10063399 | 10025710 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10063399 | 10027380 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10063399 | 10030315 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10063399 | 10031361 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10063399 | 10032546 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10063399 | 10033121 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10063399 | 10034192 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10063399 | 10034355 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10063399 | 10038703 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10063399 | 10043013 | 2 | 3 | 0 | 0 | 2 | 0 | majority_seed_inversion |
| 10063399 | 10048335 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10063399 | 10048847 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10063399 | 10051496 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10063399 | 10052254 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10063399 | 10065841 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10063399 | 10066099 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10063399 | 10068457 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10063399 | 10068961 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10063399 | 10074227 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10063399 | 10083858 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10063399 | 10084278 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10063399 | 10085505 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10063399 | 10089751 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10063399 | 10092491 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10063399 | 10099866 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10063399 | 10106168 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10063399 | 10111232 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10063399 | 10111615 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10063399 | 10124320 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10063399 | 10130640 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10063399 | 10137578 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10063399 | 10144325 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10064351 | 10001110 | 3 | 3 | 0 | 0 | 3 | 0 | all_seed_inversion |
| 10064351 | 10004992 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10064351 | 10005075 | 1 | 3 | 0 | 0 | 1 | 0 | single_seed_inversion |
| 10064351 | 10007169 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10064351 | 10007340 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10064351 | 10007452 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10064351 | 10009149 | 3 | 3 | 0 | 0 | 3 | 0 | all_seed_inversion |
| 10064351 | 10010767 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10064351 | 10013708 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10064351 | 10015079 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10064351 | 10016203 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10064351 | 10018899 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10064351 | 10019805 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10064351 | 10020343 | 1 | 3 | 0 | 0 | 1 | 0 | single_seed_inversion |
| 10064351 | 10023011 | 2 | 3 | 0 | 0 | 0 | 2 | majority_seed_inversion |
| 10064351 | 10025710 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10064351 | 10027380 | 1 | 3 | 0 | 0 | 1 | 0 | single_seed_inversion |
| 10064351 | 10030315 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10064351 | 10031361 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10064351 | 10032546 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10064351 | 10033121 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10064351 | 10034192 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10064351 | 10034355 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10064351 | 10038703 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10064351 | 10043013 | 3 | 3 | 0 | 0 | 3 | 0 | all_seed_inversion |
| 10064351 | 10048335 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10064351 | 10048847 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10064351 | 10051496 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10064351 | 10052254 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10064351 | 10065841 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10064351 | 10066099 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10064351 | 10068457 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10064351 | 10068961 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10064351 | 10074227 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10064351 | 10083858 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10064351 | 10084278 | 1 | 3 | 0 | 0 | 1 | 0 | single_seed_inversion |
| 10064351 | 10085505 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10064351 | 10089751 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10064351 | 10092491 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10064351 | 10099866 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10064351 | 10106168 | 2 | 3 | 0 | 0 | 2 | 0 | majority_seed_inversion |
| 10064351 | 10111232 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10064351 | 10111615 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10064351 | 10124320 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10064351 | 10130640 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10064351 | 10137578 | 3 | 3 | 0 | 0 | 3 | 0 | all_seed_inversion |
| 10064351 | 10144325 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10064537 | 10001110 | 3 | 3 | 0 | 0 | 3 | 0 | all_seed_inversion |
| 10064537 | 10004992 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10064537 | 10005075 | 2 | 3 | 2 | 0 | 0 | 0 | majority_seed_inversion |
| 10064537 | 10007169 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10064537 | 10007340 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10064537 | 10007452 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10064537 | 10009149 | 3 | 3 | 0 | 0 | 3 | 0 | all_seed_inversion |
| 10064537 | 10010767 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10064537 | 10013708 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10064537 | 10015079 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10064537 | 10016203 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10064537 | 10018899 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10064537 | 10019805 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10064537 | 10020343 | 2 | 3 | 1 | 1 | 1 | 0 | majority_seed_inversion |
| 10064537 | 10023011 | 2 | 3 | 2 | 0 | 0 | 0 | majority_seed_inversion |
| 10064537 | 10025710 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10064537 | 10027380 | 3 | 3 | 1 | 1 | 2 | 0 | all_seed_inversion |
| 10064537 | 10030315 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10064537 | 10031361 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10064537 | 10032546 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10064537 | 10033121 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10064537 | 10034192 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10064537 | 10034355 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10064537 | 10038703 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10064537 | 10043013 | 3 | 3 | 0 | 0 | 3 | 0 | all_seed_inversion |
| 10064537 | 10048335 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10064537 | 10048847 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10064537 | 10051496 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10064537 | 10052254 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10064537 | 10065841 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10064537 | 10066099 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10064537 | 10068457 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10064537 | 10068961 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10064537 | 10074227 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10064537 | 10083858 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10064537 | 10084278 | 2 | 3 | 1 | 0 | 1 | 0 | majority_seed_inversion |
| 10064537 | 10085505 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10064537 | 10089751 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10064537 | 10092491 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10064537 | 10099866 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10064537 | 10106168 | 2 | 3 | 1 | 1 | 1 | 0 | majority_seed_inversion |
| 10064537 | 10111232 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10064537 | 10111615 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10064537 | 10124320 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10064537 | 10130640 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10064537 | 10137578 | 3 | 3 | 2 | 2 | 1 | 0 | all_seed_inversion |
| 10064537 | 10144325 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10064626 | 10001110 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10064626 | 10004992 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10064626 | 10005075 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10064626 | 10007169 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10064626 | 10007340 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10064626 | 10007452 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10064626 | 10009149 | 2 | 3 | 0 | 0 | 2 | 0 | majority_seed_inversion |
| 10064626 | 10010767 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10064626 | 10013708 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10064626 | 10015079 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10064626 | 10016203 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10064626 | 10018899 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10064626 | 10019805 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10064626 | 10020343 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10064626 | 10023011 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10064626 | 10025710 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10064626 | 10027380 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10064626 | 10030315 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10064626 | 10031361 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10064626 | 10032546 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10064626 | 10033121 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10064626 | 10034192 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10064626 | 10034355 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10064626 | 10038703 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10064626 | 10043013 | 1 | 3 | 1 | 1 | 0 | 0 | single_seed_inversion |
| 10064626 | 10048335 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10064626 | 10048847 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10064626 | 10051496 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10064626 | 10052254 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10064626 | 10065841 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10064626 | 10066099 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10064626 | 10068457 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10064626 | 10068961 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10064626 | 10074227 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10064626 | 10083858 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10064626 | 10084278 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10064626 | 10085505 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10064626 | 10089751 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10064626 | 10092491 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10064626 | 10099866 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10064626 | 10106168 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10064626 | 10111232 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10064626 | 10111615 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10064626 | 10124320 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10064626 | 10130640 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10064626 | 10137578 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10064626 | 10144325 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10066912 | 10001110 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10066912 | 10004992 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10066912 | 10005075 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10066912 | 10007169 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10066912 | 10007340 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10066912 | 10007452 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10066912 | 10009149 | 2 | 3 | 0 | 0 | 2 | 0 | majority_seed_inversion |
| 10066912 | 10010767 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10066912 | 10013708 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10066912 | 10015079 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10066912 | 10016203 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10066912 | 10018899 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10066912 | 10019805 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10066912 | 10020343 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10066912 | 10023011 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10066912 | 10025710 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10066912 | 10027380 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10066912 | 10030315 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10066912 | 10031361 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10066912 | 10032546 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10066912 | 10033121 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10066912 | 10034192 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10066912 | 10034355 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10066912 | 10038703 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10066912 | 10043013 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10066912 | 10048335 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10066912 | 10048847 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10066912 | 10051496 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10066912 | 10052254 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10066912 | 10065841 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10066912 | 10066099 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10066912 | 10068457 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10066912 | 10068961 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10066912 | 10074227 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10066912 | 10083858 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10066912 | 10084278 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10066912 | 10085505 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10066912 | 10089751 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10066912 | 10092491 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10066912 | 10099866 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10066912 | 10106168 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10066912 | 10111232 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10066912 | 10111615 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10066912 | 10124320 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10066912 | 10130640 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10066912 | 10137578 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10066912 | 10144325 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10067220 | 10001110 | 3 | 3 | 0 | 0 | 3 | 0 | all_seed_inversion |
| 10067220 | 10004992 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10067220 | 10005075 | 1 | 3 | 1 | 0 | 0 | 0 | single_seed_inversion |
| 10067220 | 10007169 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10067220 | 10007340 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10067220 | 10007452 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10067220 | 10009149 | 3 | 3 | 0 | 0 | 3 | 0 | all_seed_inversion |
| 10067220 | 10010767 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10067220 | 10013708 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10067220 | 10015079 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10067220 | 10016203 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10067220 | 10018899 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10067220 | 10019805 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10067220 | 10020343 | 1 | 3 | 0 | 0 | 1 | 0 | single_seed_inversion |
| 10067220 | 10023011 | 2 | 3 | 0 | 0 | 0 | 2 | majority_seed_inversion |
| 10067220 | 10025710 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10067220 | 10027380 | 1 | 3 | 0 | 0 | 1 | 0 | single_seed_inversion |
| 10067220 | 10030315 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10067220 | 10031361 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10067220 | 10032546 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10067220 | 10033121 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10067220 | 10034192 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10067220 | 10034355 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10067220 | 10038703 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10067220 | 10043013 | 3 | 3 | 0 | 0 | 3 | 0 | all_seed_inversion |
| 10067220 | 10048335 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10067220 | 10048847 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10067220 | 10051496 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10067220 | 10052254 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10067220 | 10065841 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10067220 | 10066099 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10067220 | 10068457 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10067220 | 10068961 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10067220 | 10074227 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10067220 | 10083858 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10067220 | 10084278 | 1 | 3 | 0 | 0 | 1 | 0 | single_seed_inversion |
| 10067220 | 10085505 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10067220 | 10089751 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10067220 | 10092491 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10067220 | 10099866 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10067220 | 10106168 | 2 | 3 | 0 | 0 | 2 | 0 | majority_seed_inversion |
| 10067220 | 10111232 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10067220 | 10111615 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10067220 | 10124320 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10067220 | 10130640 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10067220 | 10137578 | 3 | 3 | 1 | 1 | 2 | 0 | all_seed_inversion |
| 10067220 | 10144325 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10075536 | 10001110 | 3 | 3 | 0 | 0 | 3 | 0 | all_seed_inversion |
| 10075536 | 10004992 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10075536 | 10005075 | 3 | 3 | 1 | 0 | 0 | 2 | all_seed_inversion |
| 10075536 | 10007169 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10075536 | 10007340 | 2 | 3 | 0 | 0 | 2 | 0 | majority_seed_inversion |
| 10075536 | 10007452 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10075536 | 10009149 | 3 | 3 | 0 | 0 | 3 | 0 | all_seed_inversion |
| 10075536 | 10010767 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10075536 | 10013708 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10075536 | 10015079 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10075536 | 10016203 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10075536 | 10018899 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10075536 | 10019805 | 1 | 3 | 0 | 0 | 0 | 1 | single_seed_inversion |
| 10075536 | 10020343 | 3 | 3 | 0 | 0 | 2 | 1 | all_seed_inversion |
| 10075536 | 10023011 | 3 | 3 | 0 | 0 | 0 | 3 | all_seed_inversion |
| 10075536 | 10025710 | 2 | 3 | 0 | 0 | 0 | 2 | majority_seed_inversion |
| 10075536 | 10027380 | 3 | 3 | 2 | 0 | 1 | 0 | all_seed_inversion |
| 10075536 | 10030315 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10075536 | 10031361 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10075536 | 10032546 | 3 | 3 | 2 | 0 | 0 | 1 | all_seed_inversion |
| 10075536 | 10033121 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10075536 | 10034192 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10075536 | 10034355 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10075536 | 10038703 | 2 | 3 | 0 | 0 | 0 | 2 | majority_seed_inversion |
| 10075536 | 10043013 | 3 | 3 | 0 | 0 | 3 | 0 | all_seed_inversion |
| 10075536 | 10048335 | 1 | 3 | 1 | 0 | 0 | 0 | single_seed_inversion |
| 10075536 | 10048847 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10075536 | 10051496 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10075536 | 10052254 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10075536 | 10065841 | 2 | 3 | 2 | 0 | 0 | 0 | majority_seed_inversion |
| 10075536 | 10066099 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10075536 | 10068457 | 1 | 3 | 0 | 0 | 0 | 1 | single_seed_inversion |
| 10075536 | 10068961 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10075536 | 10074227 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10075536 | 10083858 | 3 | 3 | 0 | 0 | 0 | 3 | all_seed_inversion |
| 10075536 | 10084278 | 3 | 3 | 2 | 0 | 1 | 0 | all_seed_inversion |
| 10075536 | 10085505 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10075536 | 10089751 | 1 | 3 | 0 | 0 | 0 | 1 | single_seed_inversion |
| 10075536 | 10092491 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10075536 | 10099866 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10075536 | 10106168 | 3 | 3 | 2 | 0 | 1 | 0 | all_seed_inversion |
| 10075536 | 10111232 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10075536 | 10111615 | 1 | 3 | 0 | 0 | 0 | 1 | single_seed_inversion |
| 10075536 | 10124320 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10075536 | 10130640 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10075536 | 10137578 | 3 | 3 | 3 | 0 | 0 | 0 | all_seed_inversion |
| 10075536 | 10144325 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10079757 | 10001110 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10079757 | 10004992 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10079757 | 10005075 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10079757 | 10007169 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10079757 | 10007340 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10079757 | 10007452 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10079757 | 10009149 | 1 | 3 | 0 | 0 | 1 | 0 | single_seed_inversion |
| 10079757 | 10010767 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10079757 | 10013708 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10079757 | 10015079 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10079757 | 10016203 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10079757 | 10018899 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10079757 | 10019805 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10079757 | 10020343 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10079757 | 10023011 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10079757 | 10025710 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10079757 | 10027380 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10079757 | 10030315 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10079757 | 10031361 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10079757 | 10032546 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10079757 | 10033121 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10079757 | 10034192 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10079757 | 10034355 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10079757 | 10038703 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10079757 | 10043013 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10079757 | 10048335 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10079757 | 10048847 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10079757 | 10051496 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10079757 | 10052254 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10079757 | 10065841 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10079757 | 10066099 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10079757 | 10068457 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10079757 | 10068961 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10079757 | 10074227 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10079757 | 10083858 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10079757 | 10084278 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10079757 | 10085505 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10079757 | 10089751 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10079757 | 10092491 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10079757 | 10099866 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10079757 | 10106168 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10079757 | 10111232 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10079757 | 10111615 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10079757 | 10124320 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10079757 | 10130640 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10079757 | 10137578 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10079757 | 10144325 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10082765 | 10001110 | 3 | 3 | 0 | 0 | 3 | 0 | all_seed_inversion |
| 10082765 | 10004992 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10082765 | 10005075 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10082765 | 10007169 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10082765 | 10007340 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10082765 | 10007452 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10082765 | 10009149 | 3 | 3 | 0 | 0 | 3 | 0 | all_seed_inversion |
| 10082765 | 10010767 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10082765 | 10013708 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10082765 | 10015079 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10082765 | 10016203 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10082765 | 10018899 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10082765 | 10019805 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10082765 | 10020343 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10082765 | 10023011 | 1 | 3 | 1 | 1 | 0 | 0 | single_seed_inversion |
| 10082765 | 10025710 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10082765 | 10027380 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10082765 | 10030315 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10082765 | 10031361 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10082765 | 10032546 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10082765 | 10033121 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10082765 | 10034192 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10082765 | 10034355 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10082765 | 10038703 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10082765 | 10043013 | 3 | 3 | 0 | 0 | 3 | 0 | all_seed_inversion |
| 10082765 | 10048335 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10082765 | 10048847 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10082765 | 10051496 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10082765 | 10052254 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10082765 | 10065841 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10082765 | 10066099 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10082765 | 10068457 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10082765 | 10068961 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10082765 | 10074227 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10082765 | 10083858 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10082765 | 10084278 | 1 | 3 | 1 | 1 | 0 | 0 | single_seed_inversion |
| 10082765 | 10085505 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10082765 | 10089751 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10082765 | 10092491 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10082765 | 10099866 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10082765 | 10106168 | 1 | 3 | 1 | 1 | 0 | 0 | single_seed_inversion |
| 10082765 | 10111232 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10082765 | 10111615 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10082765 | 10124320 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10082765 | 10130640 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10082765 | 10137578 | 2 | 3 | 2 | 2 | 0 | 0 | majority_seed_inversion |
| 10082765 | 10144325 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10083859 | 10001110 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10083859 | 10004992 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10083859 | 10005075 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10083859 | 10007169 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10083859 | 10007340 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10083859 | 10007452 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10083859 | 10009149 | 3 | 3 | 1 | 0 | 1 | 1 | all_seed_inversion |
| 10083859 | 10010767 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10083859 | 10013708 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10083859 | 10015079 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10083859 | 10016203 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10083859 | 10018899 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10083859 | 10019805 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10083859 | 10020343 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10083859 | 10023011 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10083859 | 10025710 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10083859 | 10027380 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10083859 | 10030315 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10083859 | 10031361 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10083859 | 10032546 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10083859 | 10033121 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10083859 | 10034192 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10083859 | 10034355 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10083859 | 10038703 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10083859 | 10043013 | 1 | 3 | 0 | 0 | 0 | 1 | single_seed_inversion |
| 10083859 | 10048335 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10083859 | 10048847 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10083859 | 10051496 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10083859 | 10052254 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10083859 | 10065841 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10083859 | 10066099 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10083859 | 10068457 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10083859 | 10068961 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10083859 | 10074227 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10083859 | 10083858 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10083859 | 10084278 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10083859 | 10085505 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10083859 | 10089751 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10083859 | 10092491 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10083859 | 10099866 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10083859 | 10106168 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10083859 | 10111232 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10083859 | 10111615 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10083859 | 10124320 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10083859 | 10130640 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10083859 | 10137578 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10083859 | 10144325 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10084329 | 10001110 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10084329 | 10004992 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10084329 | 10005075 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10084329 | 10007169 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10084329 | 10007340 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10084329 | 10007452 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10084329 | 10009149 | 3 | 3 | 0 | 0 | 3 | 0 | all_seed_inversion |
| 10084329 | 10010767 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10084329 | 10013708 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10084329 | 10015079 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10084329 | 10016203 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10084329 | 10018899 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10084329 | 10019805 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10084329 | 10020343 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10084329 | 10023011 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10084329 | 10025710 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10084329 | 10027380 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10084329 | 10030315 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10084329 | 10031361 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10084329 | 10032546 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10084329 | 10033121 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10084329 | 10034192 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10084329 | 10034355 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10084329 | 10038703 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10084329 | 10043013 | 1 | 3 | 0 | 0 | 0 | 1 | single_seed_inversion |
| 10084329 | 10048335 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10084329 | 10048847 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10084329 | 10051496 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10084329 | 10052254 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10084329 | 10065841 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10084329 | 10066099 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10084329 | 10068457 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10084329 | 10068961 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10084329 | 10074227 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10084329 | 10083858 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10084329 | 10084278 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10084329 | 10085505 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10084329 | 10089751 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10084329 | 10092491 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10084329 | 10099866 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10084329 | 10106168 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10084329 | 10111232 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10084329 | 10111615 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10084329 | 10124320 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10084329 | 10130640 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10084329 | 10137578 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10084329 | 10144325 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10091684 | 10001110 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10091684 | 10004992 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10091684 | 10005075 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10091684 | 10007169 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10091684 | 10007340 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10091684 | 10007452 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10091684 | 10009149 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10091684 | 10010767 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10091684 | 10013708 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10091684 | 10015079 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10091684 | 10016203 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10091684 | 10018899 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10091684 | 10019805 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10091684 | 10020343 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10091684 | 10023011 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10091684 | 10025710 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10091684 | 10027380 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10091684 | 10030315 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10091684 | 10031361 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10091684 | 10032546 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10091684 | 10033121 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10091684 | 10034192 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10091684 | 10034355 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10091684 | 10038703 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10091684 | 10043013 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10091684 | 10048335 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10091684 | 10048847 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10091684 | 10051496 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10091684 | 10052254 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10091684 | 10065841 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10091684 | 10066099 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10091684 | 10068457 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10091684 | 10068961 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10091684 | 10074227 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10091684 | 10083858 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10091684 | 10084278 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10091684 | 10085505 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10091684 | 10089751 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10091684 | 10092491 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10091684 | 10099866 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10091684 | 10106168 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10091684 | 10111232 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10091684 | 10111615 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10091684 | 10124320 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10091684 | 10130640 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10091684 | 10137578 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10091684 | 10144325 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10093113 | 10001110 | 2 | 3 | 0 | 0 | 2 | 0 | majority_seed_inversion |
| 10093113 | 10004992 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10093113 | 10005075 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10093113 | 10007169 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10093113 | 10007340 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10093113 | 10007452 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10093113 | 10009149 | 3 | 3 | 0 | 0 | 3 | 0 | all_seed_inversion |
| 10093113 | 10010767 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10093113 | 10013708 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10093113 | 10015079 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10093113 | 10016203 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10093113 | 10018899 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10093113 | 10019805 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10093113 | 10020343 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10093113 | 10023011 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10093113 | 10025710 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10093113 | 10027380 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10093113 | 10030315 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10093113 | 10031361 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10093113 | 10032546 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10093113 | 10033121 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10093113 | 10034192 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10093113 | 10034355 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10093113 | 10038703 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10093113 | 10043013 | 3 | 3 | 0 | 0 | 3 | 0 | all_seed_inversion |
| 10093113 | 10048335 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10093113 | 10048847 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10093113 | 10051496 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10093113 | 10052254 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10093113 | 10065841 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10093113 | 10066099 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10093113 | 10068457 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10093113 | 10068961 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10093113 | 10074227 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10093113 | 10083858 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10093113 | 10084278 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10093113 | 10085505 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10093113 | 10089751 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10093113 | 10092491 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10093113 | 10099866 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10093113 | 10106168 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10093113 | 10111232 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10093113 | 10111615 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10093113 | 10124320 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10093113 | 10130640 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10093113 | 10137578 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10093113 | 10144325 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10098069 | 10001110 | 3 | 3 | 0 | 0 | 3 | 0 | all_seed_inversion |
| 10098069 | 10004992 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10098069 | 10005075 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10098069 | 10007169 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10098069 | 10007340 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10098069 | 10007452 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10098069 | 10009149 | 3 | 3 | 0 | 0 | 3 | 0 | all_seed_inversion |
| 10098069 | 10010767 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10098069 | 10013708 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10098069 | 10015079 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10098069 | 10016203 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10098069 | 10018899 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10098069 | 10019805 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10098069 | 10020343 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10098069 | 10023011 | 1 | 3 | 0 | 0 | 0 | 1 | single_seed_inversion |
| 10098069 | 10025710 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10098069 | 10027380 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10098069 | 10030315 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10098069 | 10031361 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10098069 | 10032546 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10098069 | 10033121 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10098069 | 10034192 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10098069 | 10034355 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10098069 | 10038703 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10098069 | 10043013 | 3 | 3 | 0 | 0 | 3 | 0 | all_seed_inversion |
| 10098069 | 10048335 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10098069 | 10048847 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10098069 | 10051496 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10098069 | 10052254 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10098069 | 10065841 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10098069 | 10066099 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10098069 | 10068457 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10098069 | 10068961 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10098069 | 10074227 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10098069 | 10083858 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10098069 | 10084278 | 1 | 3 | 0 | 0 | 1 | 0 | single_seed_inversion |
| 10098069 | 10085505 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10098069 | 10089751 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10098069 | 10092491 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10098069 | 10099866 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10098069 | 10106168 | 2 | 3 | 0 | 0 | 2 | 0 | majority_seed_inversion |
| 10098069 | 10111232 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10098069 | 10111615 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10098069 | 10124320 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10098069 | 10130640 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10098069 | 10137578 | 2 | 3 | 0 | 0 | 2 | 0 | majority_seed_inversion |
| 10098069 | 10144325 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10103564 | 10001110 | 3 | 3 | 0 | 0 | 3 | 0 | all_seed_inversion |
| 10103564 | 10004992 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10103564 | 10005075 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10103564 | 10007169 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10103564 | 10007340 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10103564 | 10007452 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10103564 | 10009149 | 3 | 3 | 0 | 0 | 3 | 0 | all_seed_inversion |
| 10103564 | 10010767 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10103564 | 10013708 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10103564 | 10015079 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10103564 | 10016203 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10103564 | 10018899 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10103564 | 10019805 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10103564 | 10020343 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10103564 | 10023011 | 1 | 3 | 0 | 0 | 0 | 1 | single_seed_inversion |
| 10103564 | 10025710 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10103564 | 10027380 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10103564 | 10030315 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10103564 | 10031361 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10103564 | 10032546 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10103564 | 10033121 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10103564 | 10034192 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10103564 | 10034355 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10103564 | 10038703 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10103564 | 10043013 | 3 | 3 | 0 | 0 | 3 | 0 | all_seed_inversion |
| 10103564 | 10048335 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10103564 | 10048847 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10103564 | 10051496 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10103564 | 10052254 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10103564 | 10065841 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10103564 | 10066099 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10103564 | 10068457 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10103564 | 10068961 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10103564 | 10074227 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10103564 | 10083858 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10103564 | 10084278 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10103564 | 10085505 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10103564 | 10089751 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10103564 | 10092491 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10103564 | 10099866 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10103564 | 10106168 | 1 | 3 | 0 | 0 | 1 | 0 | single_seed_inversion |
| 10103564 | 10111232 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10103564 | 10111615 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10103564 | 10124320 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10103564 | 10130640 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10103564 | 10137578 | 1 | 3 | 0 | 0 | 1 | 0 | single_seed_inversion |
| 10103564 | 10144325 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10108395 | 10001110 | 1 | 3 | 0 | 0 | 0 | 1 | single_seed_inversion |
| 10108395 | 10004992 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10108395 | 10005075 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10108395 | 10007169 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10108395 | 10007340 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10108395 | 10007452 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10108395 | 10009149 | 3 | 3 | 0 | 0 | 3 | 0 | all_seed_inversion |
| 10108395 | 10010767 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10108395 | 10013708 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10108395 | 10015079 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10108395 | 10016203 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10108395 | 10018899 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10108395 | 10019805 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10108395 | 10020343 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10108395 | 10023011 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10108395 | 10025710 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10108395 | 10027380 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10108395 | 10030315 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10108395 | 10031361 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10108395 | 10032546 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10108395 | 10033121 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10108395 | 10034192 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10108395 | 10034355 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10108395 | 10038703 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10108395 | 10043013 | 2 | 3 | 0 | 0 | 1 | 1 | majority_seed_inversion |
| 10108395 | 10048335 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10108395 | 10048847 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10108395 | 10051496 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10108395 | 10052254 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10108395 | 10065841 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10108395 | 10066099 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10108395 | 10068457 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10108395 | 10068961 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10108395 | 10074227 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10108395 | 10083858 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10108395 | 10084278 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10108395 | 10085505 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10108395 | 10089751 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10108395 | 10092491 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10108395 | 10099866 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10108395 | 10106168 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10108395 | 10111232 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10108395 | 10111615 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10108395 | 10124320 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10108395 | 10130640 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10108395 | 10137578 | 1 | 3 | 1 | 1 | 0 | 0 | single_seed_inversion |
| 10108395 | 10144325 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10110001 | 10001110 | 3 | 3 | 0 | 0 | 3 | 0 | all_seed_inversion |
| 10110001 | 10004992 | 3 | 3 | 3 | 2 | 0 | 0 | all_seed_inversion |
| 10110001 | 10005075 | 3 | 3 | 1 | 0 | 2 | 0 | all_seed_inversion |
| 10110001 | 10007169 | 3 | 3 | 2 | 2 | 1 | 0 | all_seed_inversion |
| 10110001 | 10007340 | 3 | 3 | 0 | 0 | 3 | 0 | all_seed_inversion |
| 10110001 | 10007452 | 3 | 3 | 1 | 0 | 1 | 1 | all_seed_inversion |
| 10110001 | 10009149 | 3 | 3 | 0 | 0 | 3 | 0 | all_seed_inversion |
| 10110001 | 10010767 | 3 | 3 | 3 | 1 | 0 | 0 | all_seed_inversion |
| 10110001 | 10013708 | 3 | 3 | 3 | 0 | 0 | 0 | all_seed_inversion |
| 10110001 | 10015079 | 3 | 3 | 3 | 1 | 0 | 0 | all_seed_inversion |
| 10110001 | 10016203 | 1 | 3 | 1 | 1 | 0 | 0 | single_seed_inversion |
| 10110001 | 10018899 | 3 | 3 | 3 | 0 | 0 | 0 | all_seed_inversion |
| 10110001 | 10019805 | 3 | 3 | 2 | 0 | 1 | 0 | all_seed_inversion |
| 10110001 | 10020343 | 3 | 3 | 1 | 0 | 2 | 0 | all_seed_inversion |
| 10110001 | 10023011 | 3 | 3 | 3 | 0 | 0 | 0 | all_seed_inversion |
| 10110001 | 10025710 | 3 | 3 | 2 | 0 | 1 | 0 | all_seed_inversion |
| 10110001 | 10027380 | 3 | 3 | 1 | 0 | 2 | 0 | all_seed_inversion |
| 10110001 | 10030315 | 3 | 3 | 3 | 1 | 0 | 0 | all_seed_inversion |
| 10110001 | 10031361 | 3 | 3 | 3 | 1 | 0 | 0 | all_seed_inversion |
| 10110001 | 10032546 | 3 | 3 | 1 | 0 | 2 | 0 | all_seed_inversion |
| 10110001 | 10033121 | 3 | 3 | 3 | 2 | 0 | 0 | all_seed_inversion |
| 10110001 | 10034192 | 3 | 3 | 2 | 1 | 1 | 0 | all_seed_inversion |
| 10110001 | 10034355 | 3 | 3 | 3 | 1 | 0 | 0 | all_seed_inversion |
| 10110001 | 10038703 | 3 | 3 | 0 | 0 | 3 | 0 | all_seed_inversion |
| 10110001 | 10043013 | 3 | 3 | 0 | 0 | 3 | 0 | all_seed_inversion |
| 10110001 | 10048335 | 3 | 3 | 0 | 0 | 3 | 0 | all_seed_inversion |
| 10110001 | 10048847 | 3 | 3 | 2 | 1 | 1 | 0 | all_seed_inversion |
| 10110001 | 10051496 | 3 | 3 | 3 | 0 | 0 | 0 | all_seed_inversion |
| 10110001 | 10052254 | 3 | 3 | 3 | 2 | 0 | 0 | all_seed_inversion |
| 10110001 | 10065841 | 3 | 3 | 2 | 0 | 1 | 0 | all_seed_inversion |
| 10110001 | 10066099 | 3 | 3 | 2 | 2 | 1 | 0 | all_seed_inversion |
| 10110001 | 10068457 | 3 | 3 | 3 | 0 | 0 | 0 | all_seed_inversion |
| 10110001 | 10068961 | 3 | 3 | 3 | 1 | 0 | 0 | all_seed_inversion |
| 10110001 | 10074227 | 3 | 3 | 1 | 0 | 2 | 0 | all_seed_inversion |
| 10110001 | 10083858 | 3 | 3 | 1 | 0 | 2 | 0 | all_seed_inversion |
| 10110001 | 10084278 | 3 | 3 | 2 | 0 | 1 | 0 | all_seed_inversion |
| 10110001 | 10085505 | 3 | 3 | 3 | 2 | 0 | 0 | all_seed_inversion |
| 10110001 | 10089751 | 3 | 3 | 3 | 0 | 0 | 0 | all_seed_inversion |
| 10110001 | 10092491 | 2 | 3 | 0 | 0 | 2 | 0 | majority_seed_inversion |
| 10110001 | 10099866 | 2 | 3 | 0 | 0 | 1 | 1 | majority_seed_inversion |
| 10110001 | 10106168 | 3 | 3 | 1 | 0 | 2 | 0 | all_seed_inversion |
| 10110001 | 10111232 | 3 | 3 | 2 | 1 | 1 | 0 | all_seed_inversion |
| 10110001 | 10111615 | 3 | 3 | 1 | 0 | 2 | 0 | all_seed_inversion |
| 10110001 | 10124320 | 3 | 3 | 2 | 1 | 1 | 0 | all_seed_inversion |
| 10110001 | 10130640 | 3 | 3 | 3 | 0 | 0 | 0 | all_seed_inversion |
| 10110001 | 10137578 | 3 | 3 | 1 | 0 | 2 | 0 | all_seed_inversion |
| 10110001 | 10144325 | 2 | 3 | 2 | 2 | 0 | 0 | majority_seed_inversion |
| 10117252 | 10001110 | 3 | 3 | 0 | 0 | 3 | 0 | all_seed_inversion |
| 10117252 | 10004992 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10117252 | 10005075 | 3 | 3 | 0 | 0 | 3 | 0 | all_seed_inversion |
| 10117252 | 10007169 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10117252 | 10007340 | 2 | 3 | 0 | 0 | 2 | 0 | majority_seed_inversion |
| 10117252 | 10007452 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10117252 | 10009149 | 3 | 3 | 0 | 0 | 3 | 0 | all_seed_inversion |
| 10117252 | 10010767 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10117252 | 10013708 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10117252 | 10015079 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10117252 | 10016203 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10117252 | 10018899 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10117252 | 10019805 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10117252 | 10020343 | 3 | 3 | 0 | 0 | 2 | 1 | all_seed_inversion |
| 10117252 | 10023011 | 3 | 3 | 3 | 0 | 0 | 0 | all_seed_inversion |
| 10117252 | 10025710 | 1 | 3 | 0 | 0 | 0 | 1 | single_seed_inversion |
| 10117252 | 10027380 | 3 | 3 | 0 | 0 | 3 | 0 | all_seed_inversion |
| 10117252 | 10030315 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10117252 | 10031361 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10117252 | 10032546 | 1 | 3 | 1 | 0 | 0 | 0 | single_seed_inversion |
| 10117252 | 10033121 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10117252 | 10034192 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10117252 | 10034355 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10117252 | 10038703 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10117252 | 10043013 | 3 | 3 | 0 | 0 | 3 | 0 | all_seed_inversion |
| 10117252 | 10048335 | 1 | 3 | 0 | 0 | 1 | 0 | single_seed_inversion |
| 10117252 | 10048847 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10117252 | 10051496 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10117252 | 10052254 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10117252 | 10065841 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10117252 | 10066099 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10117252 | 10068457 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10117252 | 10068961 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10117252 | 10074227 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10117252 | 10083858 | 2 | 3 | 0 | 0 | 2 | 0 | majority_seed_inversion |
| 10117252 | 10084278 | 3 | 3 | 1 | 0 | 2 | 0 | all_seed_inversion |
| 10117252 | 10085505 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10117252 | 10089751 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10117252 | 10092491 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10117252 | 10099866 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10117252 | 10106168 | 3 | 3 | 0 | 0 | 3 | 0 | all_seed_inversion |
| 10117252 | 10111232 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10117252 | 10111615 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10117252 | 10124320 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10117252 | 10130640 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10117252 | 10137578 | 3 | 3 | 0 | 0 | 3 | 0 | all_seed_inversion |
| 10117252 | 10144325 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10119735 | 10001110 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10119735 | 10004992 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10119735 | 10005075 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10119735 | 10007169 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10119735 | 10007340 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10119735 | 10007452 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10119735 | 10009149 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10119735 | 10010767 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10119735 | 10013708 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10119735 | 10015079 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10119735 | 10016203 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10119735 | 10018899 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10119735 | 10019805 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10119735 | 10020343 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10119735 | 10023011 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10119735 | 10025710 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10119735 | 10027380 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10119735 | 10030315 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10119735 | 10031361 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10119735 | 10032546 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10119735 | 10033121 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10119735 | 10034192 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10119735 | 10034355 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10119735 | 10038703 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10119735 | 10043013 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10119735 | 10048335 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10119735 | 10048847 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10119735 | 10051496 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10119735 | 10052254 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10119735 | 10065841 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10119735 | 10066099 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10119735 | 10068457 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10119735 | 10068961 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10119735 | 10074227 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10119735 | 10083858 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10119735 | 10084278 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10119735 | 10085505 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10119735 | 10089751 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10119735 | 10092491 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10119735 | 10099866 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10119735 | 10106168 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10119735 | 10111232 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10119735 | 10111615 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10119735 | 10124320 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10119735 | 10130640 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10119735 | 10137578 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10119735 | 10144325 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10124993 | 10001110 | 3 | 3 | 0 | 0 | 0 | 3 | all_seed_inversion |
| 10124993 | 10004992 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10124993 | 10005075 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10124993 | 10007169 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10124993 | 10007340 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10124993 | 10007452 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10124993 | 10009149 | 3 | 3 | 2 | 0 | 0 | 1 | all_seed_inversion |
| 10124993 | 10010767 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10124993 | 10013708 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10124993 | 10015079 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10124993 | 10016203 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10124993 | 10018899 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10124993 | 10019805 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10124993 | 10020343 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10124993 | 10023011 | 1 | 3 | 0 | 0 | 0 | 1 | single_seed_inversion |
| 10124993 | 10025710 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10124993 | 10027380 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10124993 | 10030315 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10124993 | 10031361 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10124993 | 10032546 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10124993 | 10033121 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10124993 | 10034192 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10124993 | 10034355 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10124993 | 10038703 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10124993 | 10043013 | 3 | 3 | 0 | 0 | 0 | 3 | all_seed_inversion |
| 10124993 | 10048335 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10124993 | 10048847 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10124993 | 10051496 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10124993 | 10052254 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10124993 | 10065841 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10124993 | 10066099 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10124993 | 10068457 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10124993 | 10068961 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10124993 | 10074227 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10124993 | 10083858 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10124993 | 10084278 | 1 | 3 | 1 | 0 | 0 | 0 | single_seed_inversion |
| 10124993 | 10085505 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10124993 | 10089751 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10124993 | 10092491 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10124993 | 10099866 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10124993 | 10106168 | 1 | 3 | 1 | 0 | 0 | 0 | single_seed_inversion |
| 10124993 | 10111232 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10124993 | 10111615 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10124993 | 10124320 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10124993 | 10130640 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10124993 | 10137578 | 2 | 3 | 2 | 1 | 0 | 0 | majority_seed_inversion |
| 10124993 | 10144325 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10127720 | 10001110 | 3 | 3 | 0 | 0 | 3 | 0 | all_seed_inversion |
| 10127720 | 10004992 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10127720 | 10005075 | 3 | 3 | 0 | 0 | 0 | 3 | all_seed_inversion |
| 10127720 | 10007169 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10127720 | 10007340 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10127720 | 10007452 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10127720 | 10009149 | 3 | 3 | 0 | 0 | 3 | 0 | all_seed_inversion |
| 10127720 | 10010767 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10127720 | 10013708 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10127720 | 10015079 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10127720 | 10016203 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10127720 | 10018899 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10127720 | 10019805 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10127720 | 10020343 | 2 | 3 | 0 | 0 | 1 | 1 | majority_seed_inversion |
| 10127720 | 10023011 | 3 | 3 | 2 | 0 | 0 | 1 | all_seed_inversion |
| 10127720 | 10025710 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10127720 | 10027380 | 3 | 3 | 3 | 0 | 0 | 0 | all_seed_inversion |
| 10127720 | 10030315 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10127720 | 10031361 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10127720 | 10032546 | 1 | 3 | 0 | 0 | 0 | 1 | single_seed_inversion |
| 10127720 | 10033121 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10127720 | 10034192 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10127720 | 10034355 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10127720 | 10038703 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10127720 | 10043013 | 3 | 3 | 0 | 0 | 3 | 0 | all_seed_inversion |
| 10127720 | 10048335 | 1 | 3 | 1 | 0 | 0 | 0 | single_seed_inversion |
| 10127720 | 10048847 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10127720 | 10051496 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10127720 | 10052254 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10127720 | 10065841 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10127720 | 10066099 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10127720 | 10068457 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10127720 | 10068961 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10127720 | 10074227 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10127720 | 10083858 | 1 | 3 | 0 | 0 | 0 | 1 | single_seed_inversion |
| 10127720 | 10084278 | 2 | 3 | 2 | 0 | 0 | 0 | majority_seed_inversion |
| 10127720 | 10085505 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10127720 | 10089751 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10127720 | 10092491 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10127720 | 10099866 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10127720 | 10106168 | 3 | 3 | 2 | 1 | 1 | 0 | all_seed_inversion |
| 10127720 | 10111232 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10127720 | 10111615 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10127720 | 10124320 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10127720 | 10130640 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10127720 | 10137578 | 3 | 3 | 2 | 0 | 1 | 0 | all_seed_inversion |
| 10127720 | 10144325 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10131359 | 10001110 | 3 | 3 | 0 | 0 | 3 | 0 | all_seed_inversion |
| 10131359 | 10004992 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10131359 | 10005075 | 1 | 3 | 0 | 0 | 0 | 1 | single_seed_inversion |
| 10131359 | 10007169 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10131359 | 10007340 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10131359 | 10007452 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10131359 | 10009149 | 3 | 3 | 0 | 0 | 3 | 0 | all_seed_inversion |
| 10131359 | 10010767 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10131359 | 10013708 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10131359 | 10015079 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10131359 | 10016203 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10131359 | 10018899 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10131359 | 10019805 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10131359 | 10020343 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10131359 | 10023011 | 2 | 3 | 0 | 0 | 0 | 2 | majority_seed_inversion |
| 10131359 | 10025710 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10131359 | 10027380 | 2 | 3 | 0 | 0 | 2 | 0 | majority_seed_inversion |
| 10131359 | 10030315 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10131359 | 10031361 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10131359 | 10032546 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10131359 | 10033121 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10131359 | 10034192 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10131359 | 10034355 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10131359 | 10038703 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10131359 | 10043013 | 3 | 3 | 0 | 0 | 3 | 0 | all_seed_inversion |
| 10131359 | 10048335 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10131359 | 10048847 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10131359 | 10051496 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10131359 | 10052254 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10131359 | 10065841 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10131359 | 10066099 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10131359 | 10068457 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10131359 | 10068961 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10131359 | 10074227 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10131359 | 10083858 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10131359 | 10084278 | 2 | 3 | 1 | 0 | 1 | 0 | majority_seed_inversion |
| 10131359 | 10085505 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10131359 | 10089751 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10131359 | 10092491 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10131359 | 10099866 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10131359 | 10106168 | 2 | 3 | 1 | 1 | 1 | 0 | majority_seed_inversion |
| 10131359 | 10111232 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10131359 | 10111615 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10131359 | 10124320 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10131359 | 10130640 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10131359 | 10137578 | 3 | 3 | 2 | 1 | 1 | 0 | all_seed_inversion |
| 10131359 | 10144325 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10132146 | 10001110 | 3 | 3 | 0 | 0 | 3 | 0 | all_seed_inversion |
| 10132146 | 10004992 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10132146 | 10005075 | 3 | 3 | 0 | 0 | 1 | 2 | all_seed_inversion |
| 10132146 | 10007169 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10132146 | 10007340 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10132146 | 10007452 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10132146 | 10009149 | 3 | 3 | 0 | 0 | 3 | 0 | all_seed_inversion |
| 10132146 | 10010767 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10132146 | 10013708 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10132146 | 10015079 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10132146 | 10016203 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10132146 | 10018899 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10132146 | 10019805 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10132146 | 10020343 | 2 | 3 | 0 | 0 | 1 | 1 | majority_seed_inversion |
| 10132146 | 10023011 | 2 | 3 | 0 | 0 | 0 | 2 | majority_seed_inversion |
| 10132146 | 10025710 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10132146 | 10027380 | 3 | 3 | 1 | 1 | 2 | 0 | all_seed_inversion |
| 10132146 | 10030315 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10132146 | 10031361 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10132146 | 10032546 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10132146 | 10033121 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10132146 | 10034192 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10132146 | 10034355 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10132146 | 10038703 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10132146 | 10043013 | 3 | 3 | 0 | 0 | 3 | 0 | all_seed_inversion |
| 10132146 | 10048335 | 1 | 3 | 1 | 1 | 0 | 0 | single_seed_inversion |
| 10132146 | 10048847 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10132146 | 10051496 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10132146 | 10052254 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10132146 | 10065841 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10132146 | 10066099 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10132146 | 10068457 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10132146 | 10068961 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10132146 | 10074227 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10132146 | 10083858 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10132146 | 10084278 | 3 | 3 | 1 | 0 | 2 | 0 | all_seed_inversion |
| 10132146 | 10085505 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10132146 | 10089751 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10132146 | 10092491 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10132146 | 10099866 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10132146 | 10106168 | 3 | 3 | 1 | 1 | 2 | 0 | all_seed_inversion |
| 10132146 | 10111232 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10132146 | 10111615 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10132146 | 10124320 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10132146 | 10130640 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10132146 | 10137578 | 3 | 3 | 1 | 0 | 2 | 0 | all_seed_inversion |
| 10132146 | 10144325 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10132330 | 10001110 | 2 | 3 | 0 | 0 | 1 | 1 | majority_seed_inversion |
| 10132330 | 10004992 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10132330 | 10005075 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10132330 | 10007169 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10132330 | 10007340 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10132330 | 10007452 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10132330 | 10009149 | 3 | 3 | 0 | 0 | 3 | 0 | all_seed_inversion |
| 10132330 | 10010767 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10132330 | 10013708 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10132330 | 10015079 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10132330 | 10016203 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10132330 | 10018899 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10132330 | 10019805 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10132330 | 10020343 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10132330 | 10023011 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10132330 | 10025710 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10132330 | 10027380 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10132330 | 10030315 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10132330 | 10031361 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10132330 | 10032546 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10132330 | 10033121 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10132330 | 10034192 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10132330 | 10034355 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10132330 | 10038703 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10132330 | 10043013 | 3 | 3 | 0 | 0 | 3 | 0 | all_seed_inversion |
| 10132330 | 10048335 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10132330 | 10048847 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10132330 | 10051496 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10132330 | 10052254 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10132330 | 10065841 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10132330 | 10066099 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10132330 | 10068457 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10132330 | 10068961 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10132330 | 10074227 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10132330 | 10083858 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10132330 | 10084278 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10132330 | 10085505 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10132330 | 10089751 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10132330 | 10092491 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10132330 | 10099866 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10132330 | 10106168 | 1 | 3 | 0 | 0 | 0 | 0 | single_seed_inversion |
| 10132330 | 10111232 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10132330 | 10111615 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10132330 | 10124320 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10132330 | 10130640 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10132330 | 10137578 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10132330 | 10144325 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10135408 | 10001110 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10135408 | 10004992 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10135408 | 10005075 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10135408 | 10007169 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10135408 | 10007340 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10135408 | 10007452 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10135408 | 10009149 | 1 | 3 | 0 | 0 | 1 | 0 | single_seed_inversion |
| 10135408 | 10010767 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10135408 | 10013708 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10135408 | 10015079 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10135408 | 10016203 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10135408 | 10018899 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10135408 | 10019805 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10135408 | 10020343 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10135408 | 10023011 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10135408 | 10025710 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10135408 | 10027380 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10135408 | 10030315 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10135408 | 10031361 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10135408 | 10032546 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10135408 | 10033121 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10135408 | 10034192 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10135408 | 10034355 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10135408 | 10038703 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10135408 | 10043013 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10135408 | 10048335 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10135408 | 10048847 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10135408 | 10051496 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10135408 | 10052254 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10135408 | 10065841 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10135408 | 10066099 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10135408 | 10068457 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10135408 | 10068961 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10135408 | 10074227 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10135408 | 10083858 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10135408 | 10084278 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10135408 | 10085505 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10135408 | 10089751 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10135408 | 10092491 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10135408 | 10099866 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10135408 | 10106168 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10135408 | 10111232 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10135408 | 10111615 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10135408 | 10124320 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10135408 | 10130640 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10135408 | 10137578 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10135408 | 10144325 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10138067 | 10001110 | 2 | 3 | 0 | 0 | 2 | 0 | majority_seed_inversion |
| 10138067 | 10004992 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10138067 | 10005075 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10138067 | 10007169 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10138067 | 10007340 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10138067 | 10007452 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10138067 | 10009149 | 3 | 3 | 0 | 0 | 3 | 0 | all_seed_inversion |
| 10138067 | 10010767 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10138067 | 10013708 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10138067 | 10015079 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10138067 | 10016203 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10138067 | 10018899 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10138067 | 10019805 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10138067 | 10020343 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10138067 | 10023011 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10138067 | 10025710 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10138067 | 10027380 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10138067 | 10030315 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10138067 | 10031361 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10138067 | 10032546 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10138067 | 10033121 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10138067 | 10034192 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10138067 | 10034355 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10138067 | 10038703 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10138067 | 10043013 | 3 | 3 | 0 | 0 | 3 | 0 | all_seed_inversion |
| 10138067 | 10048335 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10138067 | 10048847 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10138067 | 10051496 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10138067 | 10052254 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10138067 | 10065841 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10138067 | 10066099 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10138067 | 10068457 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10138067 | 10068961 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10138067 | 10074227 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10138067 | 10083858 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10138067 | 10084278 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10138067 | 10085505 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10138067 | 10089751 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10138067 | 10092491 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10138067 | 10099866 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10138067 | 10106168 | 1 | 3 | 0 | 0 | 1 | 0 | single_seed_inversion |
| 10138067 | 10111232 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10138067 | 10111615 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10138067 | 10124320 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10138067 | 10130640 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10138067 | 10137578 | 1 | 3 | 1 | 1 | 0 | 0 | single_seed_inversion |
| 10138067 | 10144325 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10138528 | 10001110 | 3 | 3 | 0 | 0 | 3 | 0 | all_seed_inversion |
| 10138528 | 10004992 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10138528 | 10005075 | 1 | 3 | 1 | 0 | 0 | 0 | single_seed_inversion |
| 10138528 | 10007169 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10138528 | 10007340 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10138528 | 10007452 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10138528 | 10009149 | 3 | 3 | 0 | 0 | 3 | 0 | all_seed_inversion |
| 10138528 | 10010767 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10138528 | 10013708 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10138528 | 10015079 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10138528 | 10016203 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10138528 | 10018899 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10138528 | 10019805 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10138528 | 10020343 | 1 | 3 | 0 | 0 | 1 | 0 | single_seed_inversion |
| 10138528 | 10023011 | 2 | 3 | 2 | 0 | 0 | 0 | majority_seed_inversion |
| 10138528 | 10025710 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10138528 | 10027380 | 1 | 3 | 1 | 0 | 0 | 0 | single_seed_inversion |
| 10138528 | 10030315 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10138528 | 10031361 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10138528 | 10032546 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10138528 | 10033121 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10138528 | 10034192 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10138528 | 10034355 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10138528 | 10038703 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10138528 | 10043013 | 3 | 3 | 0 | 0 | 3 | 0 | all_seed_inversion |
| 10138528 | 10048335 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10138528 | 10048847 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10138528 | 10051496 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10138528 | 10052254 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10138528 | 10065841 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10138528 | 10066099 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10138528 | 10068457 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10138528 | 10068961 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10138528 | 10074227 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10138528 | 10083858 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10138528 | 10084278 | 1 | 3 | 1 | 0 | 0 | 0 | single_seed_inversion |
| 10138528 | 10085505 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10138528 | 10089751 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10138528 | 10092491 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10138528 | 10099866 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10138528 | 10106168 | 2 | 3 | 2 | 0 | 0 | 0 | majority_seed_inversion |
| 10138528 | 10111232 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10138528 | 10111615 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10138528 | 10124320 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10138528 | 10130640 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10138528 | 10137578 | 2 | 3 | 1 | 0 | 1 | 0 | majority_seed_inversion |
| 10138528 | 10144325 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10145328 | 10001110 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10145328 | 10004992 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10145328 | 10005075 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10145328 | 10007169 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10145328 | 10007340 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10145328 | 10007452 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10145328 | 10009149 | 3 | 3 | 2 | 0 | 1 | 0 | all_seed_inversion |
| 10145328 | 10010767 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10145328 | 10013708 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10145328 | 10015079 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10145328 | 10016203 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10145328 | 10018899 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10145328 | 10019805 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10145328 | 10020343 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10145328 | 10023011 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10145328 | 10025710 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10145328 | 10027380 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10145328 | 10030315 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10145328 | 10031361 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10145328 | 10032546 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10145328 | 10033121 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10145328 | 10034192 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10145328 | 10034355 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10145328 | 10038703 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10145328 | 10043013 | 1 | 3 | 0 | 0 | 0 | 1 | single_seed_inversion |
| 10145328 | 10048335 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10145328 | 10048847 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10145328 | 10051496 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10145328 | 10052254 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10145328 | 10065841 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10145328 | 10066099 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10145328 | 10068457 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10145328 | 10068961 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10145328 | 10074227 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10145328 | 10083858 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10145328 | 10084278 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10145328 | 10085505 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10145328 | 10089751 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10145328 | 10092491 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10145328 | 10099866 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10145328 | 10106168 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10145328 | 10111232 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10145328 | 10111615 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10145328 | 10124320 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10145328 | 10130640 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10145328 | 10137578 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10145328 | 10144325 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10151451 | 10001110 | 3 | 3 | 0 | 0 | 3 | 0 | all_seed_inversion |
| 10151451 | 10004992 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10151451 | 10005075 | 1 | 3 | 0 | 0 | 1 | 0 | single_seed_inversion |
| 10151451 | 10007169 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10151451 | 10007340 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10151451 | 10007452 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10151451 | 10009149 | 3 | 3 | 0 | 0 | 3 | 0 | all_seed_inversion |
| 10151451 | 10010767 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10151451 | 10013708 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10151451 | 10015079 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10151451 | 10016203 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10151451 | 10018899 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10151451 | 10019805 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10151451 | 10020343 | 1 | 3 | 0 | 0 | 1 | 0 | single_seed_inversion |
| 10151451 | 10023011 | 2 | 3 | 0 | 0 | 0 | 2 | majority_seed_inversion |
| 10151451 | 10025710 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10151451 | 10027380 | 1 | 3 | 0 | 0 | 1 | 0 | single_seed_inversion |
| 10151451 | 10030315 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10151451 | 10031361 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10151451 | 10032546 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10151451 | 10033121 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10151451 | 10034192 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10151451 | 10034355 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10151451 | 10038703 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10151451 | 10043013 | 3 | 3 | 0 | 0 | 3 | 0 | all_seed_inversion |
| 10151451 | 10048335 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10151451 | 10048847 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10151451 | 10051496 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10151451 | 10052254 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10151451 | 10065841 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10151451 | 10066099 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10151451 | 10068457 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10151451 | 10068961 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10151451 | 10074227 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10151451 | 10083858 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10151451 | 10084278 | 1 | 3 | 0 | 0 | 1 | 0 | single_seed_inversion |
| 10151451 | 10085505 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10151451 | 10089751 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10151451 | 10092491 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10151451 | 10099866 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10151451 | 10106168 | 2 | 3 | 0 | 0 | 2 | 0 | majority_seed_inversion |
| 10151451 | 10111232 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10151451 | 10111615 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10151451 | 10124320 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10151451 | 10130640 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10151451 | 10137578 | 3 | 3 | 0 | 0 | 3 | 0 | all_seed_inversion |
| 10151451 | 10144325 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10151887 | 10001110 | 3 | 3 | 0 | 0 | 3 | 0 | all_seed_inversion |
| 10151887 | 10004992 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10151887 | 10005075 | 3 | 3 | 0 | 0 | 3 | 0 | all_seed_inversion |
| 10151887 | 10007169 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10151887 | 10007340 | 2 | 3 | 0 | 0 | 2 | 0 | majority_seed_inversion |
| 10151887 | 10007452 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10151887 | 10009149 | 3 | 3 | 0 | 0 | 3 | 0 | all_seed_inversion |
| 10151887 | 10010767 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10151887 | 10013708 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10151887 | 10015079 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10151887 | 10016203 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10151887 | 10018899 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10151887 | 10019805 | 1 | 3 | 0 | 0 | 1 | 0 | single_seed_inversion |
| 10151887 | 10020343 | 3 | 3 | 0 | 0 | 3 | 0 | all_seed_inversion |
| 10151887 | 10023011 | 3 | 3 | 0 | 0 | 1 | 2 | all_seed_inversion |
| 10151887 | 10025710 | 1 | 3 | 0 | 0 | 1 | 0 | single_seed_inversion |
| 10151887 | 10027380 | 3 | 3 | 0 | 0 | 3 | 0 | all_seed_inversion |
| 10151887 | 10030315 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10151887 | 10031361 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10151887 | 10032546 | 2 | 3 | 0 | 0 | 2 | 0 | majority_seed_inversion |
| 10151887 | 10033121 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10151887 | 10034192 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10151887 | 10034355 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10151887 | 10038703 | 2 | 3 | 0 | 0 | 2 | 0 | majority_seed_inversion |
| 10151887 | 10043013 | 3 | 3 | 0 | 0 | 3 | 0 | all_seed_inversion |
| 10151887 | 10048335 | 1 | 3 | 0 | 0 | 1 | 0 | single_seed_inversion |
| 10151887 | 10048847 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10151887 | 10051496 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10151887 | 10052254 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10151887 | 10065841 | 2 | 3 | 0 | 0 | 1 | 1 | majority_seed_inversion |
| 10151887 | 10066099 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10151887 | 10068457 | 1 | 3 | 0 | 0 | 1 | 0 | single_seed_inversion |
| 10151887 | 10068961 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10151887 | 10074227 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10151887 | 10083858 | 3 | 3 | 0 | 0 | 3 | 0 | all_seed_inversion |
| 10151887 | 10084278 | 3 | 3 | 0 | 0 | 2 | 1 | all_seed_inversion |
| 10151887 | 10085505 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10151887 | 10089751 | 1 | 3 | 0 | 0 | 1 | 0 | single_seed_inversion |
| 10151887 | 10092491 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10151887 | 10099866 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10151887 | 10106168 | 3 | 3 | 0 | 0 | 3 | 0 | all_seed_inversion |
| 10151887 | 10111232 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10151887 | 10111615 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10151887 | 10124320 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10151887 | 10130640 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10151887 | 10137578 | 3 | 3 | 0 | 0 | 3 | 0 | all_seed_inversion |
| 10151887 | 10144325 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10156499 | 10001110 | 3 | 3 | 0 | 0 | 3 | 0 | all_seed_inversion |
| 10156499 | 10004992 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10156499 | 10005075 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10156499 | 10007169 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10156499 | 10007340 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10156499 | 10007452 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10156499 | 10009149 | 3 | 3 | 0 | 0 | 3 | 0 | all_seed_inversion |
| 10156499 | 10010767 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10156499 | 10013708 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10156499 | 10015079 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10156499 | 10016203 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10156499 | 10018899 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10156499 | 10019805 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10156499 | 10020343 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10156499 | 10023011 | 1 | 3 | 1 | 1 | 0 | 0 | single_seed_inversion |
| 10156499 | 10025710 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10156499 | 10027380 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10156499 | 10030315 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10156499 | 10031361 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10156499 | 10032546 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10156499 | 10033121 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10156499 | 10034192 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10156499 | 10034355 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10156499 | 10038703 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10156499 | 10043013 | 3 | 3 | 0 | 0 | 3 | 0 | all_seed_inversion |
| 10156499 | 10048335 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10156499 | 10048847 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10156499 | 10051496 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10156499 | 10052254 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10156499 | 10065841 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10156499 | 10066099 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10156499 | 10068457 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10156499 | 10068961 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10156499 | 10074227 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10156499 | 10083858 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10156499 | 10084278 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10156499 | 10085505 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10156499 | 10089751 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10156499 | 10092491 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10156499 | 10099866 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10156499 | 10106168 | 1 | 3 | 1 | 1 | 0 | 0 | single_seed_inversion |
| 10156499 | 10111232 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10156499 | 10111615 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10156499 | 10124320 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10156499 | 10130640 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10156499 | 10137578 | 1 | 3 | 0 | 0 | 1 | 0 | single_seed_inversion |
| 10156499 | 10144325 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10157441 | 10001110 | 3 | 3 | 0 | 0 | 3 | 0 | all_seed_inversion |
| 10157441 | 10004992 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10157441 | 10005075 | 3 | 3 | 2 | 0 | 1 | 0 | all_seed_inversion |
| 10157441 | 10007169 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10157441 | 10007340 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10157441 | 10007452 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10157441 | 10009149 | 3 | 3 | 0 | 0 | 3 | 0 | all_seed_inversion |
| 10157441 | 10010767 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10157441 | 10013708 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10157441 | 10015079 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10157441 | 10016203 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10157441 | 10018899 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10157441 | 10019805 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10157441 | 10020343 | 2 | 3 | 1 | 0 | 1 | 0 | majority_seed_inversion |
| 10157441 | 10023011 | 2 | 3 | 2 | 0 | 0 | 0 | majority_seed_inversion |
| 10157441 | 10025710 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10157441 | 10027380 | 3 | 3 | 1 | 1 | 2 | 0 | all_seed_inversion |
| 10157441 | 10030315 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10157441 | 10031361 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10157441 | 10032546 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10157441 | 10033121 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10157441 | 10034192 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10157441 | 10034355 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10157441 | 10038703 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10157441 | 10043013 | 3 | 3 | 0 | 0 | 3 | 0 | all_seed_inversion |
| 10157441 | 10048335 | 1 | 3 | 0 | 0 | 1 | 0 | single_seed_inversion |
| 10157441 | 10048847 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10157441 | 10051496 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10157441 | 10052254 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10157441 | 10065841 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10157441 | 10066099 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10157441 | 10068457 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10157441 | 10068961 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10157441 | 10074227 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10157441 | 10083858 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10157441 | 10084278 | 2 | 3 | 2 | 0 | 0 | 0 | majority_seed_inversion |
| 10157441 | 10085505 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10157441 | 10089751 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10157441 | 10092491 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10157441 | 10099866 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10157441 | 10106168 | 3 | 3 | 1 | 1 | 2 | 0 | all_seed_inversion |
| 10157441 | 10111232 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10157441 | 10111615 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10157441 | 10124320 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10157441 | 10130640 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10157441 | 10137578 | 3 | 3 | 2 | 2 | 1 | 0 | all_seed_inversion |
| 10157441 | 10144325 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10163258 | 10001110 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10163258 | 10004992 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10163258 | 10005075 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10163258 | 10007169 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10163258 | 10007340 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10163258 | 10007452 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10163258 | 10009149 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10163258 | 10010767 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10163258 | 10013708 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10163258 | 10015079 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10163258 | 10016203 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10163258 | 10018899 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10163258 | 10019805 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10163258 | 10020343 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10163258 | 10023011 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10163258 | 10025710 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10163258 | 10027380 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10163258 | 10030315 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10163258 | 10031361 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10163258 | 10032546 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10163258 | 10033121 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10163258 | 10034192 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10163258 | 10034355 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10163258 | 10038703 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10163258 | 10043013 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10163258 | 10048335 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10163258 | 10048847 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10163258 | 10051496 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10163258 | 10052254 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10163258 | 10065841 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10163258 | 10066099 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10163258 | 10068457 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10163258 | 10068961 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10163258 | 10074227 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10163258 | 10083858 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10163258 | 10084278 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10163258 | 10085505 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10163258 | 10089751 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10163258 | 10092491 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10163258 | 10099866 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10163258 | 10106168 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10163258 | 10111232 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10163258 | 10111615 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10163258 | 10124320 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10163258 | 10130640 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10163258 | 10137578 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10163258 | 10144325 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10164295 | 10001110 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10164295 | 10004992 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10164295 | 10005075 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10164295 | 10007169 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10164295 | 10007340 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10164295 | 10007452 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10164295 | 10009149 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10164295 | 10010767 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10164295 | 10013708 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10164295 | 10015079 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10164295 | 10016203 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10164295 | 10018899 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10164295 | 10019805 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10164295 | 10020343 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10164295 | 10023011 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10164295 | 10025710 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10164295 | 10027380 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10164295 | 10030315 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10164295 | 10031361 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10164295 | 10032546 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10164295 | 10033121 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10164295 | 10034192 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10164295 | 10034355 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10164295 | 10038703 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10164295 | 10043013 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10164295 | 10048335 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10164295 | 10048847 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10164295 | 10051496 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10164295 | 10052254 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10164295 | 10065841 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10164295 | 10066099 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10164295 | 10068457 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10164295 | 10068961 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10164295 | 10074227 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10164295 | 10083858 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10164295 | 10084278 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10164295 | 10085505 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10164295 | 10089751 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10164295 | 10092491 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10164295 | 10099866 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10164295 | 10106168 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10164295 | 10111232 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10164295 | 10111615 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10164295 | 10124320 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10164295 | 10130640 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10164295 | 10137578 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10164295 | 10144325 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10168610 | 10001110 | 2 | 3 | 0 | 0 | 1 | 1 | majority_seed_inversion |
| 10168610 | 10004992 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10168610 | 10005075 | 2 | 3 | 0 | 0 | 0 | 2 | majority_seed_inversion |
| 10168610 | 10007169 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10168610 | 10007340 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10168610 | 10007452 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10168610 | 10009149 | 3 | 3 | 0 | 0 | 2 | 1 | all_seed_inversion |
| 10168610 | 10010767 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10168610 | 10013708 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10168610 | 10015079 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10168610 | 10016203 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10168610 | 10018899 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10168610 | 10019805 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10168610 | 10020343 | 1 | 3 | 0 | 0 | 1 | 0 | single_seed_inversion |
| 10168610 | 10023011 | 2 | 3 | 0 | 0 | 0 | 2 | majority_seed_inversion |
| 10168610 | 10025710 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10168610 | 10027380 | 2 | 3 | 0 | 0 | 0 | 2 | majority_seed_inversion |
| 10168610 | 10030315 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10168610 | 10031361 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10168610 | 10032546 | 1 | 3 | 0 | 0 | 0 | 1 | single_seed_inversion |
| 10168610 | 10033121 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10168610 | 10034192 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10168610 | 10034355 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10168610 | 10038703 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10168610 | 10043013 | 3 | 3 | 0 | 0 | 2 | 1 | all_seed_inversion |
| 10168610 | 10048335 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10168610 | 10048847 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10168610 | 10051496 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10168610 | 10052254 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10168610 | 10065841 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10168610 | 10066099 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10168610 | 10068457 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10168610 | 10068961 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10168610 | 10074227 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10168610 | 10083858 | 1 | 3 | 0 | 0 | 0 | 1 | single_seed_inversion |
| 10168610 | 10084278 | 2 | 3 | 0 | 0 | 0 | 2 | majority_seed_inversion |
| 10168610 | 10085505 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10168610 | 10089751 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10168610 | 10092491 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10168610 | 10099866 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10168610 | 10106168 | 2 | 3 | 0 | 0 | 0 | 2 | majority_seed_inversion |
| 10168610 | 10111232 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10168610 | 10111615 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10168610 | 10124320 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10168610 | 10130640 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |
| 10168610 | 10137578 | 2 | 3 | 0 | 0 | 0 | 2 | majority_seed_inversion |
| 10168610 | 10144325 | 0 | 3 | 0 | 0 | 0 | 0 | not_inversion_or_partial |

## Failure Flags

| seed | positive_patient_id | negative_patient_id | positive_logit | negative_logit | final_logit_margin | is_inversion | positive_pred_prob | negative_pred_prob | positive_text_contribution | negative_text_contribution | text_margin | positive_image_contribution | negative_image_contribution | image_margin | positive_bio_contribution | negative_bio_contribution | bio_margin | positive_fusion_contribution | negative_fusion_contribution | fusion_margin | positive_discordance_contribution | negative_discordance_contribution | discordance_margin | margin_without_image | margin_without_text | margin_without_bio | text_only_like_margin | image_only_like_margin | margin_without_diffuse | margin_without_negative | image_opposed_flag | image_repair_flag | text_driven_flag | text_strong_flag | fusion_interaction_flag |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 10168610 | 10009149 | 0.15747468173503876 | 0.7076072692871094 | -0.5501325875520706 | 1 | 0.5392875671386719 | 0.6698722839355469 | 0.042086053639650345 | 0.19372443854808807 | -0.15163838490843773 | 0.2035103440284729 | -1.0245403051376343 | 1.2280506491661072 | -0.1310800164937973 | -0.42696183919906616 | 0.29588182270526886 | 0.04295830801129341 | 1.9653849601745605 | -1.9224266521632671 | unavailable | unavailable | unavailable | -1.6009143888950348 | 1.3493564128875732 | -1.5406432151794434 | -2.473503455519676 | 0.9363256692886353 | 1.5024016797542572 | -0.7310405075550079 | 0 | 0 | 1 | 1 | 0 |
| 0 | 10168610 | 10043013 | 0.15747468173503876 | 0.6961038112640381 | -0.5386291295289993 | 1 | 0.5392875671386719 | 0.6673234105110168 | 0.042086053639650345 | 0.1282225400209427 | -0.08613648638129234 | 0.2035103440284729 | -1.2837620973587036 | 1.4872724413871765 | -0.1310800164937973 | -0.38241881132125854 | 0.25133879482746124 | 0.04295830801129341 | 2.2340621948242188 | -2.1911038868129253 | unavailable | unavailable | unavailable | -2.2530485689640045 | 1.538272500038147 | -1.6798955202102661 | -3.478259429335594 | 0.9493870735168457 | 1.1909516751766205 | -0.34533432126045227 | 0 | 0 | 1 | 1 | 0 |
| 0 | 10012205 | 10048335 | -0.8558571934700012 | -0.463720440864563 | -0.39213675260543823 | 1 | 0.2982056140899658 | 0.3861036002635956 | -0.012109387665987015 | 0.033699337393045425 | -0.04580872505903244 | -1.1763969659805298 | -1.0547784566879272 | -0.12161850929260254 | -0.5049495697021484 | -0.13584637641906738 | -0.36910319328308105 | 0.8375987410545349 | 0.6932049989700317 | 0.14439374208450317 | unavailable | unavailable | unavailable | -0.25566810369491577 | -0.6371966600418091 | 0.09549635648727417 | 0.21100908517837524 | -0.18771231174468994 | -0.3919113874435425 | -0.5120993852615356 | 0 | 0 | 1 | 0 | 0 |
| 0 | 10012205 | 10001110 | -0.8558571934700012 | 0.15342950820922852 | -1.0092867016792297 | 1 | 0.2982056140899658 | 0.5382822751998901 | -0.012109387665987015 | 0.1399565190076828 | -0.15206590667366982 | -1.1763969659805298 | -1.14687979221344 | -0.029517173767089844 | -0.5049495697021484 | -0.4019477963447571 | -0.10300177335739136 | 0.8375987410545349 | 1.562300682067871 | -0.7247019410133362 | unavailable | unavailable | unavailable | -0.8872326016426086 | -0.26448774337768555 | -1.1149965524673462 | -0.945711612701416 | -0.15698862075805664 | 0.712622344493866 | -1.9177908897399902 | 0 | 0 | 1 | 0 | 0 |
| 0 | 10012205 | 10009149 | -0.8558571934700012 | 0.7076072692871094 | -1.5634644627571106 | 1 | 0.2982056140899658 | 0.6698722839355469 | -0.012109387665987015 | 0.19372443854808807 | -0.2058338262140751 | -1.1763969659805298 | -1.0245403051376343 | -0.1518566608428955 | -0.5049495697021484 | -0.42696183919906616 | -0.07798773050308228 | 0.8375987410545349 | 1.9653849601745605 | -1.1277862191200256 | unavailable | unavailable | unavailable | -1.3293092846870422 | -0.30591869354248047 | -1.798589050769806 | -1.5144580602645874 | -0.17237448692321777 | 0.6200478672981262 | -2.47351336479187 | 0 | 0 | 1 | 0 | 0 |
| 0 | 10012205 | 10043013 | -0.8558571934700012 | 0.6961038112640381 | -1.5519610047340393 | 1 | 0.2982056140899658 | 0.6673234105110168 | -0.012109387665987015 | 0.1282225400209427 | -0.1403319276869297 | -1.1763969659805298 | -1.2837620973587036 | 0.10736513137817383 | -0.5049495697021484 | -0.38241881132125854 | -0.12253075838088989 | 0.8375987410545349 | 2.2340621948242188 | -1.3964634537696838 | unavailable | unavailable | unavailable | -1.981443464756012 | -0.11700260639190674 | -1.9378413558006287 | -2.5192140340805054 | -0.15931308269500732 | 0.3085978627204895 | -2.0878071784973145 | 0 | 0 | 1 | 1 | 0 |
| 0 | 10012205 | 10106168 | -0.8558571934700012 | -0.4428281784057617 | -0.4130290150642395 | 1 | 0.2982056140899658 | 0.3910672664642334 | -0.012109387665987015 | 0.015184640884399414 | -0.02729402855038643 | -1.1763969659805298 | -0.8689271211624146 | -0.30746984481811523 | -0.5049495697021484 | 0.18552681803703308 | -0.6904763877391815 | 0.8375987410545349 | 0.22538746893405914 | 0.6122112721204758 | unavailable | unavailable | unavailable | -0.1525018811225891 | -1.047685295343399 | 0.5731240510940552 | 0.8646384850144386 | -0.33256101608276367 | -0.15095847845077515 | -1.2837526202201843 | 0 | 0 | 1 | 0 | 0 |
| 0 | 10012205 | 10027380 | -0.8558571934700012 | -0.3543927073478699 | -0.5014644861221313 | 1 | 0.2982056140899658 | 0.4123176038265228 | -0.012109387665987015 | 0.02382062003016472 | -0.03593000769615173 | -1.1763969659805298 | -0.8366108536720276 | -0.3397861123085022 | -0.5049495697021484 | -0.37764397263526917 | -0.12730559706687927 | 0.8375987410545349 | 0.8360415101051331 | 0.0015572309494018555 | unavailable | unavailable | unavailable | -0.27757757902145386 | -0.3913114070892334 | -0.4111001491546631 | -0.214025616645813 | -0.27188825607299805 | 0.6854321360588074 | -1.4935868382453918 | 0 | 0 | 1 | 0 | 0 |
| 0 | 10012205 | 10020343 | -0.8558571934700012 | -0.35937297344207764 | -0.4964842200279236 | 1 | 0.2982056140899658 | 0.41111135482788086 | -0.012109387665987015 | -0.04554690793156624 | 0.033437520265579224 | -1.1763969659805298 | -1.2528825998306274 | 0.07648563385009766 | -0.5049495697021484 | -0.3646604120731354 | -0.14028915762901306 | 0.8375987410545349 | 1.3037168979644775 | -0.4661181569099426 | unavailable | unavailable | unavailable | -0.8121767640113831 | -0.06204378604888916 | -0.5869585871696472 | -0.9965027570724487 | 0.044001221656799316 | -0.46855801343917847 | -1.0987266898155212 | 0 | 0 | 0 | 0 | 1 |
| 0 | 10012205 | 10023011 | -0.8558571934700012 | 0.09009921550750732 | -0.9459564089775085 | 1 | 0.2982056140899658 | 0.5225095748901367 | -0.012109387665987015 | -0.07374109327793121 | 0.0616317056119442 | -1.1763969659805298 | -1.1518511772155762 | -0.024545788764953613 | -0.5049495697021484 | -0.4184007942676544 | -0.08654877543449402 | 0.8375987410545349 | 1.7340922355651855 | -0.8964934945106506 | unavailable | unavailable | unavailable | -1.4106071591377258 | -0.23278796672821045 | -1.2181275486946106 | -1.9274028539657593 | -0.4175213575363159 | -0.930946409702301 | -0.1471858024597168 | 1 | 0 | 0 | 0 | 0 |
| 0 | 10012205 | 10084278 | -0.8558571934700012 | -0.2953658103942871 | -0.5604913830757141 | 1 | 0.2982056140899658 | 0.4266907274723053 | -0.012109387665987015 | -0.07336916029453278 | 0.06125977262854576 | -1.1763969659805298 | -0.9316350221633911 | -0.24476194381713867 | -0.5049495697021484 | -0.5517039895057678 | 0.046754419803619385 | 0.8375987410545349 | 1.2613422870635986 | -0.4237435460090637 | unavailable | unavailable | unavailable | -0.9614190459251404 | -0.18098270893096924 | -0.7901345491409302 | -1.4051066637039185 | -0.5143824815750122 | 0.621554434299469 | -0.2385646104812622 | 1 | 0 | 0 | 0 | 0 |
| 0 | 10012205 | 10005075 | -0.8558571934700012 | -0.4597259759902954 | -0.3961312174797058 | 1 | 0.2982056140899658 | 0.3870508074760437 | -0.012109387665987015 | 0.030231382697820663 | -0.04234077036380768 | -1.1763969659805298 | -1.4041030406951904 | 0.22770607471466064 | -0.5049495697021484 | -0.3593677878379822 | -0.14558178186416626 | 0.8375987410545349 | 1.2735134363174438 | -0.43591469526290894 | unavailable | unavailable | unavailable | -1.357518494129181 | 0.25514674186706543 | -0.3976230025291443 | -1.7066367864608765 | 0.16983795166015625 | 1.7957951426506042 | 0.08778297901153564 | 0 | 0 | 1 | 1 | 0 |
| 0 | 10012205 | 10137578 | -0.8558571934700012 | 0.1183476448059082 | -0.9742048382759094 | 1 | 0.2982056140899658 | 0.5295524001121521 | -0.012109387665987015 | 0.05747392401099205 | -0.06958331167697906 | -1.1763969659805298 | -0.7802721858024597 | -0.39612478017807007 | -0.5049495697021484 | -0.25496262311935425 | -0.2499869465827942 | 0.8375987410545349 | 1.0961085557937622 | -0.2585098147392273 | unavailable | unavailable | unavailable | -0.5872973203659058 | -0.9017167687416077 | -0.9176264405250549 | -0.5721902847290039 | -0.5068486332893372 | -0.2833346128463745 | -1.8294867277145386 | 0 | 0 | 1 | 0 | 0 |
| 0 | 10138528 | 10001110 | -0.04569554328918457 | 0.15342950820922852 | -0.19912505149841309 | 1 | 0.48857808113098145 | 0.5382822751998901 | 0.03216021880507469 | 0.1399565190076828 | -0.10779630020260811 | -1.2209007740020752 | -1.14687979221344 | -0.07402098178863525 | 0.0715789645910263 | -0.4019477963447571 | 0.4735267609357834 | 1.071466088294983 | 1.562300682067871 | -0.4908345937728882 | unavailable | unavailable | unavailable | -0.08636474609375 | 0.47579002380371094 | -0.7772114872932434 | -0.6205117702484131 | -0.08602213859558105 | 0.41443920135498047 | -0.22612076997756958 | 0 | 0 | 1 | 0 | 0 |
| 0 | 10138528 | 10009149 | -0.04569554328918457 | 0.7076072692871094 | -0.753302812576294 | 1 | 0.48857808113098145 | 0.6698722839355469 | 0.03216021880507469 | 0.19372443854808807 | -0.16156421974301338 | -1.2209007740020752 | -1.0245403051376343 | -0.19636046886444092 | 0.0715789645910263 | -0.42696183919906616 | 0.49854080379009247 | 1.071466088294983 | 1.9653849601745605 | -0.8939188718795776 | unavailable | unavailable | unavailable | -0.5284414291381836 | 0.434359073638916 | -1.4608039855957031 | -1.1892582178115845 | -0.10140800476074219 | 0.3218647241592407 | -0.7818432450294495 | 0 | 0 | 1 | 0 | 0 |
| 0 | 10138528 | 10043013 | -0.04569554328918457 | 0.6961038112640381 | -0.7417993545532227 | 1 | 0.48857808113098145 | 0.6673234105110168 | 0.03216021880507469 | 0.1282225400209427 | -0.096062321215868 | -1.2209007740020752 | -1.2837620973587036 | 0.06286132335662842 | 0.0715789645910263 | -0.38241881132125854 | 0.45399777591228485 | 1.071466088294983 | 2.2340621948242188 | -1.1625961065292358 | unavailable | unavailable | unavailable | -1.1805756092071533 | 0.6232751607894897 | -1.6000562906265259 | -2.1940141916275024 | -0.08834660053253174 | 0.010414719581604004 | -0.3961370587348938 | 0 | 0 | 1 | 1 | 0 |
| 0 | 10138528 | 10023011 | -0.04569554328918457 | 0.09009921550750732 | -0.1357947587966919 | 1 | 0.48857808113098145 | 0.5225095748901367 | 0.03216021880507469 | -0.07374109327793121 | 0.1059013120830059 | -1.2209007740020752 | -1.1518511772155762 | -0.06904959678649902 | 0.0715789645910263 | -0.4184007942676544 | 0.4899797588586807 | 1.071466088294983 | 1.7340922355651855 | -0.6626261472702026 | unavailable | unavailable | unavailable | -0.6097393035888672 | 0.507489800453186 | -0.8803424835205078 | -1.6022030115127563 | -0.34655487537384033 | -1.2291295528411865 | 1.5444843173027039 | 1 | 0 | 0 | 0 | 0 |
| 0 | 10138528 | 10137578 | -0.04569554328918457 | 0.1183476448059082 | -0.16404318809509277 | 1 | 0.48857808113098145 | 0.5295524001121521 | 0.03216021880507469 | 0.05747392401099205 | -0.02531370520591736 | -1.2209007740020752 | -0.7802721858024597 | -0.4406285881996155 | 0.0715789645910263 | -0.25496262311935425 | 0.32654158771038055 | 1.071466088294983 | 1.0961085557937622 | -0.024642467498779297 | unavailable | unavailable | unavailable | 0.21357053518295288 | -0.16143900156021118 | -0.5798413753509521 | -0.24699044227600098 | -0.4358821511268616 | -0.58151775598526 | -0.13781660795211792 | 0 | 0 | 1 | 0 | 0 |
| 0 | 10151887 | 10048335 | -1.2415703535079956 | -0.463720440864563 | -0.7778499126434326 | 1 | 0.224162757396698 | 0.3861036002635956 | -0.06955688446760178 | 0.033699337393045425 | -0.1032562218606472 | -0.5393540859222412 | -1.0547784566879272 | 0.515424370765686 | -0.4139614701271057 | -0.13584637641906738 | -0.27811509370803833 | -0.21869787573814392 | 0.6932049989700317 | -0.9119028747081757 | unavailable | unavailable | unavailable | -1.3334161043167114 | 0.15045678615570068 | -0.6867620944976807 | -1.3290581107139587 | 0.5278534293174744 | -0.6589662432670593 | -0.039647459983825684 | 0 | 0 | 1 | 1 | 0 |
| 0 | 10151887 | 10001110 | -1.2415703535079956 | 0.15342950820922852 | -1.3949998617172241 | 1 | 0.224162757396698 | 0.5382822751998901 | -0.06955688446760178 | 0.1399565190076828 | -0.20951340347528458 | -0.5393540859222412 | -1.14687979221344 | 0.6075257062911987 | -0.4139614701271057 | -0.4019477963447571 | -0.012013673782348633 | -0.21869787573814392 | 1.562300682067871 | -1.780998557806015 | unavailable | unavailable | unavailable | -1.9649806022644043 | 0.5231657028198242 | -1.897255003452301 | -2.48577880859375 | 0.5585771203041077 | 0.4455674886703491 | -1.4453389644622803 | 0 | 0 | 1 | 1 | 0 |

## Counterfactual Margins

| seed | positive_patient_id | negative_patient_id | final_logit_margin | margin_without_image | margin_without_text | margin_without_bio | text_only_like_margin | image_only_like_margin | margin_without_diffuse | margin_without_negative | image_opposed_flag | image_repair_flag | text_driven_flag | fusion_interaction_flag |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 10168610 | 10009149 | -0.5501325875520706 | -1.6009143888950348 | 1.3493564128875732 | -1.5406432151794434 | -2.473503455519676 | 0.9363256692886353 | 1.5024016797542572 | -0.7310405075550079 | 0 | 0 | 1 | 0 |
| 0 | 10168610 | 10043013 | -0.5386291295289993 | -2.2530485689640045 | 1.538272500038147 | -1.6798955202102661 | -3.478259429335594 | 0.9493870735168457 | 1.1909516751766205 | -0.34533432126045227 | 0 | 0 | 1 | 0 |
| 0 | 10012205 | 10048335 | -0.39213675260543823 | -0.25566810369491577 | -0.6371966600418091 | 0.09549635648727417 | 0.21100908517837524 | -0.18771231174468994 | -0.3919113874435425 | -0.5120993852615356 | 0 | 0 | 1 | 0 |
| 0 | 10012205 | 10001110 | -1.0092867016792297 | -0.8872326016426086 | -0.26448774337768555 | -1.1149965524673462 | -0.945711612701416 | -0.15698862075805664 | 0.712622344493866 | -1.9177908897399902 | 0 | 0 | 1 | 0 |
| 0 | 10012205 | 10009149 | -1.5634644627571106 | -1.3293092846870422 | -0.30591869354248047 | -1.798589050769806 | -1.5144580602645874 | -0.17237448692321777 | 0.6200478672981262 | -2.47351336479187 | 0 | 0 | 1 | 0 |
| 0 | 10012205 | 10043013 | -1.5519610047340393 | -1.981443464756012 | -0.11700260639190674 | -1.9378413558006287 | -2.5192140340805054 | -0.15931308269500732 | 0.3085978627204895 | -2.0878071784973145 | 0 | 0 | 1 | 0 |
| 0 | 10012205 | 10106168 | -0.4130290150642395 | -0.1525018811225891 | -1.047685295343399 | 0.5731240510940552 | 0.8646384850144386 | -0.33256101608276367 | -0.15095847845077515 | -1.2837526202201843 | 0 | 0 | 1 | 0 |
| 0 | 10012205 | 10027380 | -0.5014644861221313 | -0.27757757902145386 | -0.3913114070892334 | -0.4111001491546631 | -0.214025616645813 | -0.27188825607299805 | 0.6854321360588074 | -1.4935868382453918 | 0 | 0 | 1 | 0 |
| 0 | 10012205 | 10020343 | -0.4964842200279236 | -0.8121767640113831 | -0.06204378604888916 | -0.5869585871696472 | -0.9965027570724487 | 0.044001221656799316 | -0.46855801343917847 | -1.0987266898155212 | 0 | 0 | 0 | 1 |
| 0 | 10012205 | 10023011 | -0.9459564089775085 | -1.4106071591377258 | -0.23278796672821045 | -1.2181275486946106 | -1.9274028539657593 | -0.4175213575363159 | -0.930946409702301 | -0.1471858024597168 | 1 | 0 | 0 | 0 |
| 0 | 10012205 | 10084278 | -0.5604913830757141 | -0.9614190459251404 | -0.18098270893096924 | -0.7901345491409302 | -1.4051066637039185 | -0.5143824815750122 | 0.621554434299469 | -0.2385646104812622 | 1 | 0 | 0 | 0 |
| 0 | 10012205 | 10005075 | -0.3961312174797058 | -1.357518494129181 | 0.25514674186706543 | -0.3976230025291443 | -1.7066367864608765 | 0.16983795166015625 | 1.7957951426506042 | 0.08778297901153564 | 0 | 0 | 1 | 0 |
| 0 | 10012205 | 10137578 | -0.9742048382759094 | -0.5872973203659058 | -0.9017167687416077 | -0.9176264405250549 | -0.5721902847290039 | -0.5068486332893372 | -0.2833346128463745 | -1.8294867277145386 | 0 | 0 | 1 | 0 |
| 0 | 10138528 | 10001110 | -0.19912505149841309 | -0.08636474609375 | 0.47579002380371094 | -0.7772114872932434 | -0.6205117702484131 | -0.08602213859558105 | 0.41443920135498047 | -0.22612076997756958 | 0 | 0 | 1 | 0 |
| 0 | 10138528 | 10009149 | -0.753302812576294 | -0.5284414291381836 | 0.434359073638916 | -1.4608039855957031 | -1.1892582178115845 | -0.10140800476074219 | 0.3218647241592407 | -0.7818432450294495 | 0 | 0 | 1 | 0 |
| 0 | 10138528 | 10043013 | -0.7417993545532227 | -1.1805756092071533 | 0.6232751607894897 | -1.6000562906265259 | -2.1940141916275024 | -0.08834660053253174 | 0.010414719581604004 | -0.3961370587348938 | 0 | 0 | 1 | 0 |
| 0 | 10138528 | 10023011 | -0.1357947587966919 | -0.6097393035888672 | 0.507489800453186 | -0.8803424835205078 | -1.6022030115127563 | -0.34655487537384033 | -1.2291295528411865 | 1.5444843173027039 | 1 | 0 | 0 | 0 |
| 0 | 10138528 | 10137578 | -0.16404318809509277 | 0.21357053518295288 | -0.16143900156021118 | -0.5798413753509521 | -0.24699044227600098 | -0.4358821511268616 | -0.58151775598526 | -0.13781660795211792 | 0 | 0 | 1 | 0 |
| 0 | 10151887 | 10048335 | -0.7778499126434326 | -1.3334161043167114 | 0.15045678615570068 | -0.6867620944976807 | -1.3290581107139587 | 0.5278534293174744 | -0.6589662432670593 | -0.039647459983825684 | 0 | 0 | 1 | 0 |
| 0 | 10151887 | 10001110 | -1.3949998617172241 | -1.9649806022644043 | 0.5231657028198242 | -1.897255003452301 | -2.48577880859375 | 0.5585771203041077 | 0.4455674886703491 | -1.4453389644622803 | 0 | 0 | 1 | 0 |

## Hard Patient Subgroups

| patient_id | inversion_count | role | inversion_share | all_seed_hard_patient | n_seeds_with_inversion | top5_share_context | top10_share_context |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 10009149 | 118 | negative | 0.13333333333333333 | 1 | 3 | 0.5932203389830508 | 0.9028248587570622 |
| 10043013 | 98 | negative | 0.11073446327683616 | 1 | 3 | 0.5932203389830508 | 0.9028248587570622 |
| 10001110 | 83 | negative | 0.09378531073446328 | 1 | 3 | 0.5932203389830508 | 0.9028248587570622 |
| 10137578 | 66 | negative | 0.07457627118644068 | 1 | 3 | 0.5932203389830508 | 0.9028248587570622 |
| 10106168 | 59 | negative | 0.06666666666666667 | 1 | 3 | 0.5932203389830508 | 0.9028248587570622 |
| 10023011 | 53 | negative | 0.059887005649717516 | 1 | 3 | 0.5932203389830508 | 0.9028248587570622 |
| 10027380 | 47 | negative | 0.05310734463276836 | 1 | 3 | 0.5932203389830508 | 0.9028248587570622 |
| 10084278 | 45 | negative | 0.05084745762711865 | 1 | 3 | 0.5932203389830508 | 0.9028248587570622 |
| 10005075 | 44 | negative | 0.04971751412429379 | 1 | 3 | 0.5932203389830508 | 0.9028248587570622 |
| 10020343 | 38 | negative | 0.04293785310734463 | 1 | 3 | 0.5932203389830508 | 0.9028248587570622 |
| 10083858 | 21 | negative | 0.023728813559322035 | 1 | 3 | 0.5932203389830508 | 0.9028248587570622 |
| 10032546 | 18 | negative | 0.020338983050847456 | 1 | 3 | 0.5932203389830508 | 0.9028248587570622 |
| 10048335 | 16 | negative | 0.01807909604519774 | 1 | 3 | 0.5932203389830508 | 0.9028248587570622 |
| 10007340 | 13 | negative | 0.014689265536723164 | 1 | 3 | 0.5932203389830508 | 0.9028248587570622 |
| 10065841 | 13 | negative | 0.014689265536723164 | 1 | 3 | 0.5932203389830508 | 0.9028248587570622 |
| 10025710 | 11 | negative | 0.012429378531073447 | 1 | 3 | 0.5932203389830508 | 0.9028248587570622 |
| 10038703 | 11 | negative | 0.012429378531073447 | 1 | 3 | 0.5932203389830508 | 0.9028248587570622 |
| 10019805 | 8 | negative | 0.00903954802259887 | 1 | 3 | 0.5932203389830508 | 0.9028248587570622 |
| 10068457 | 8 | negative | 0.00903954802259887 | 1 | 3 | 0.5932203389830508 | 0.9028248587570622 |
| 10089751 | 8 | negative | 0.00903954802259887 | 1 | 3 | 0.5932203389830508 | 0.9028248587570622 |
| 10111615 | 7 | negative | 0.007909604519774011 | 1 | 3 | 0.5932203389830508 | 0.9028248587570622 |
| 10013708 | 6 | negative | 0.006779661016949152 | 1 | 3 | 0.5932203389830508 | 0.9028248587570622 |
| 10034355 | 6 | negative | 0.006779661016949152 | 1 | 3 | 0.5932203389830508 | 0.9028248587570622 |
| 10051496 | 6 | negative | 0.006779661016949152 | 1 | 3 | 0.5932203389830508 | 0.9028248587570622 |
| 10074227 | 6 | negative | 0.006779661016949152 | 1 | 3 | 0.5932203389830508 | 0.9028248587570622 |
| 10111232 | 6 | negative | 0.006779661016949152 | 1 | 3 | 0.5932203389830508 | 0.9028248587570622 |
| 10015079 | 5 | negative | 0.005649717514124294 | 1 | 3 | 0.5932203389830508 | 0.9028248587570622 |
| 10018899 | 5 | negative | 0.005649717514124294 | 1 | 3 | 0.5932203389830508 | 0.9028248587570622 |
| 10030315 | 5 | negative | 0.005649717514124294 | 1 | 3 | 0.5932203389830508 | 0.9028248587570622 |
| 10004992 | 4 | negative | 0.004519774011299435 | 1 | 3 | 0.5932203389830508 | 0.9028248587570622 |

## Route Gate

| route | final_status | allowed_next_step | c15_authorized | reproduction_pass_all_seeds | total_pairwise_rows | total_inversion_rows | all_seed_inversion_pairs | majority_seed_inversion_pairs | single_seed_inversion_pairs | decision_basis |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| HARD_PATIENT_SUBGROUP_FAILURE | C14C_HARD_SUBGROUP_STOP | MORE_ANALYSIS_ONLY | 0 | 1 | 6627 | 885 | 215 | 75 | 90 | {"fusion_interaction_fraction": 0.08553459119496855, "fusion_margin_means": [-0.7262771141070586, -1.0607832556997892, -0.8567424396246565], "fusion_margin_negative_seed_count": 3, "image_margin_means": [-0.1641098655617008, -0.11839184983546147, -0.12391641076427207], "image_margin_negative_seed_count": 3, "image_opposed_fraction": 0.28553459119496855, "image_repair_means": [-0.7960348729903881, -1.0353188159060664, -0.901845560065307], "image_repair_positive_seed_count": 0, "majority_inversion_pairs": 75, "stable_inversion_pairs": 215, "text_driven_fraction": 0.6289308176100629, "text_margin_means": [-0.035670436381434016, -0.005998677930620033, -0.03332760452716795], "text_margin_nonpositive_seed_count": 3, "top5_patient_inversion_share": 0.5932203389830508} |

Route: `HARD_PATIENT_SUBGROUP_FAILURE`.
Final C14-C status: `C14C_HARD_SUBGROUP_STOP`.
C15 authorized: `False`.
Allowed next-step class: `MORE_ANALYSIS_ONLY`.
Gate basis: `{"fusion_interaction_fraction": 0.08553459119496855, "fusion_margin_means": [-0.7262771141070586, -1.0607832556997892, -0.8567424396246565], "fusion_margin_negative_seed_count": 3, "image_margin_means": [-0.1641098655617008, -0.11839184983546147, -0.12391641076427207], "image_margin_negative_seed_count": 3, "image_opposed_fraction": 0.28553459119496855, "image_repair_means": [-0.7960348729903881, -1.0353188159060664, -0.901845560065307], "image_repair_positive_seed_count": 0, "majority_inversion_pairs": 75, "stable_inversion_pairs": 215, "text_driven_fraction": 0.6289308176100629, "text_margin_means": [-0.035670436381434016, -0.005998677930620033, -0.03332760452716795], "text_margin_nonpositive_seed_count": 3, "top5_patient_inversion_share": 0.5932203389830508}`.

C13 remains the current strict best. C15 may run only when the route gate authorizes it; otherwise the autonomous workflow stops without training.
