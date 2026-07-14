# C26-SM Fixed Topology

| frozen source | stable slot | relation parameter |
|---|---|---|
| masked mean of all five image evidence slots | M1 morphology | scalar `image_morphology` |
| masked mean of text support and nonspecific slots | M1 morphology | scalar `text_morphology` |
| bio immune observed | M2 immune | scalar `bio_immune` |
| bio function observed | M3 function | scalar `bio_function` |
| text opposition | M4 opposition | scalar `text_opposition` |
| text temporal | M5 temporal | scalar `text_temporal` |

Frozen context order: text-global projection, then bio-other projection addition, then frozen C17 disease norm.
No mechanism-to-mechanism edge is present.
