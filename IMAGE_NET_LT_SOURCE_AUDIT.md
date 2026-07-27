# ImageNet-LT source audit

The default one-command route reconstructs the conventional ImageNet-LT
layout rather than substituting a different long-tail dataset.

| Payload | Source pinned in the runner | Integrity check | Verified contract |
| --- | --- | --- | --- |
| Original images | `https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_train.tar` | MD5 `1d675b47d978889d74fa0da5fadfb00e` | Runner expands the 1,000 synset archives to `train/<synset>/*`. |
| LT train manifest | `Vanint/SADE-AgnosticLT@cba11b8b0fb91711eeffd5e45311f321f8a88680` | SHA-256 `efdbdad4f050237c310b2f354cf95a8b1d7c8d57a63c4ea4bb6bf2bcb012f37f` | 115,846 rows; 1,000 classes; 5–1,280 images/class. |
| LT balanced reference | same revision, `ImageNet_LT_val.txt` | SHA-256 `9af7ba688acf9532ff5845a8fb14a1e97fdc4c51ac48bc72c08ea9c3f7ff142e` | 20,000 rows; exactly 20 images for each of 1,000 classes. |

The canonical OLTR repository distributes the ImageNet-LT split files
separately and expects the original ImageNet images to be downloaded first.
The pinned public manifest mirror above has the canonical row counts and is
downloaded only when the local files are absent. The launch preflight verifies
every referenced image and rejects any mismatch before a GPU task starts.

`LTX_IMAGENET_SOURCE=custom_archive` remains available only for an outage or
private mirror. It requires an explicit SHA-256 and must expand to the same
`train/<synset>/*` layout; it does not weaken the data contract.
