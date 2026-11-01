# Example photos

Five unmodified JPEGs copied from the project's existing test dataset at commit `11d56db391ac13343caaec9971bf8e7a6192e71f`. Each is a distinct dish. These are interactive examples, not a new validation set or an accuracy benchmark.

| Bundled file | Source |
| --- | --- |
| `adobo.jpg` | [dataset/test/adobo/adobo_005a1b723dc6.jpg](https://github.com/PANHARO/FoodImage/blob/11d56db391ac13343caaec9971bf8e7a6192e71f/dataset/test/adobo/adobo_005a1b723dc6.jpg) |
| `amok_trey.jpg` | [dataset/test/amok_trey/amok_trey_0cec490fc350.jpg](https://github.com/PANHARO/FoodImage/blob/11d56db391ac13343caaec9971bf8e7a6192e71f/dataset/test/amok_trey/amok_trey_0cec490fc350.jpg) |
| `banh_mi.jpg` | [dataset/test/banh_mi/banh_mi_04dcce90d1c0.jpg](https://github.com/PANHARO/FoodImage/blob/11d56db391ac13343caaec9971bf8e7a6192e71f/dataset/test/banh_mi/banh_mi_04dcce90d1c0.jpg) |
| `pho.jpg` | [dataset/test/pho/pho_11cac3ba73ba.jpg](https://github.com/PANHARO/FoodImage/blob/11d56db391ac13343caaec9971bf8e7a6192e71f/dataset/test/pho/pho_11cac3ba73ba.jpg) |
| `satay.jpg` | [dataset/test/satay/satay_0939ef1be225.jpg](https://github.com/PANHARO/FoodImage/blob/11d56db391ac13343caaec9971bf8e7a6192e71f/dataset/test/satay/satay_0939ef1be225.jpg) |

These photographs come from the owner's supplied project dataset. The repository does not provide individual photographer credits or image-license metadata. Retain any original credits you have; replace a photo with one you own if needed. No new license is asserted for these assets.

To replace examples, save RGB JPGs using the same five filenames. To change the selection, edit `EXAMPLE_NAMES` in `app.py`; each name must exist in `CLASS_NAMES` and have a matching `examples/<name>.jpg`.
