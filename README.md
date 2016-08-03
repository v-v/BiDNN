BiDNN
=====
Bidirectional (Symmetrical) Deep Neural Networks

As described in: V. Vukotić, C. Raymond, and G. Gravier. Bidirectional Joint Representation Learning with Symmetrical Deep Neural Networks for Multimodal and Crossmodal Applications. In *Proceedings of the 2016 ACM on International Conference on Multimedia Retrieval (ICMR)*, pages 343–346. ACM, 2016. 

BiDNNs are similar to (or a variation of) multimodal autoencoders. They offer crossmodal translation and superior multimodal embedding created in a common representation created by the two crossmodal translation. The following picture illustrates the architectural differences of classical multimodal autoencoders and BiDNNs:

![Alt text](images/AEvsBiDNN.png?raw=true "Title")

Multimodal Autoencoders
-----------------------

Classical multimodal autoencoders usually come in two varieties: i) standard autoencoders that are used in a multimodal setup by contactenating the modalities and ii) multimodal autoencoders that have separate layers for each modality and one or more central fully connected layers that yield a multimodal representation and serve as a translation layer (as illustrated in the left (*a*) part of the previous figure). One modality is often sporadically removed (zeroed) while a reconstruction of both modalities is expected at its output to force the multimodal autoencoder to represent one modality from the other.

Autoencoders however have some downsides which slightly deteriorate performance:
* Both modalities influence the same central layer(s), either directly or indirectly, through other modality-specific fully connected layers. Even when translating from one modality to the other, the input modality is either mixed with the other or with a zeroed input.
* Autoencoders need to learn to reconstruct the same output both when one modality is marked missing (e.g., zeroed) and when both modalities are presented as input.
* Classical autoencoders are primarily made for multimodal embedding while crossmodal translation is offered as a secondary function.

Bidirectional (Symmetrical) Deep Neural Networks
------------------------------------------------

In bidirectional deep neural networks, learning is performed in both directions: one modality is presented as an input and the other as the expected output while at the same time the second one is presented as input and the first one as expected output. This is equivalent to using two separate deep neural networks and tying them (sharing specific weight variables) to make them symmetrical, as illustrated in the right part of the previous figure (*b*). Implementation-wise the variables representing the weights are shared across the two networks and are in fact the same variables. Learning of the two crossmodal mappings is then performed simultaneously and they are forced to be as close as possible to each other's inverses by the symmetric architecture in the middle. A joint representation in the middle of the two crossmodal mappings is also formed while learning.

Given such an architecture, crossmodal translation is done straightforwardly by presenting the first modality and obtaining the output in the representation space of the second modality. Multimodal embeddings are obtained in the following manner:
* When the two modalities are available (automatic transcripts and visual concepts or CNN features, depending on the setup), both are presented at their respective inputs and the activations are propagated through the network. The multimodal embedding is then obtained by concatenating the outputs of the middle layer.
* When one modality is available and the other is not (either only transcripts or only visual information), the available modality is presented to its respective input of the network and the activations are propagated. The central layer is then used to generate an embedding by being duplicated, thus still generating an embedding of the same size while allowing to transparently compare video segments regardless of modality availability (either with only one or both modalities).

Usage
=====



