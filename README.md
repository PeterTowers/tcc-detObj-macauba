# tcc-detObj-macauba
Code related to my undergraduate thesis in the field of machine learning
and object detection. Since most of the work was done in portuguese,
the documentation will follow on this language. Most (if not all) of the
code is in english, though. If you're interested, you may contact me
through LinkedIn.

## Abstract
The macauba palm (*Acrocomia intumenscens*) has potential to become a
high-value resource within the framework of Brazil's sustainable economy.
This study evaluates two convolutional neural networks (CNN)
architectures for the tasks of identifying and differentiating
specimens of macauba and babassu (*Attalea speciosa*) in their natural
environments in the context of object detection and UAV-based imagery.
The evaluated CNN architectures were YOLOv4 and YOLOv9, which achieved
an mAP score of $75.63\% \pm 3.29\%$ (average $\pm$ standard deviation)
and $72.7\% \pm 2.3\%$, respectively, as evaluated through k-fold
cross-validation. The results indicate the effectiveness of the proposed
methodology in distinguishing these species, although there are
improvements to be made in the image annotation process, or the process
of dividing the orthomosaics that produced the images used in the
training phase.
---
## Imagens e *labels*
As imagens e labels estão contidos na pasta
`img/imagensTCC-Embrapa_500.zip`.