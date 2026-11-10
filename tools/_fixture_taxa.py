"""Etiquetas reales de SpeciesNet 4.0.2a usadas en los fixtures.

Copiadas tal cual del archivo ``always_crop_99710272_22x8_v12_epoch_00148.labels.20251208.txt``
del modelo, para que los fixtures se puedan regenerar sin tener el modelo descargado.
Los nombres comunes vienen del modelo, en inglés.
"""

JAGUAR = "9b7fea59-cfa3-499c-a420-bcf36277dcd8;mammalia;carnivora;felidae;panthera;onca;jaguar"
PUMA = "9c564562-9429-405c-8529-04cff7752282;mammalia;carnivora;felidae;puma;concolor;puma"
OCELOT = "22976d14-d424-4f18-a67a-d8e1689cefcc;mammalia;carnivora;felidae;leopardus;pardalis;ocelot"
TAPIR = "39e5ea73-38f1-454a-8054-7c67f20fbced;mammalia;perissodactyla;tapiridae;tapirus;terrestris;lowland tapir"
DASYPROCTA_GENUS = "4e6eb99a-baf0-4235-a7bc-1dd7b741b7c3;mammalia;rodentia;dasyproctidae;dasyprocta;;dasyprocta species"
AGOUTI = "8bc77c4f-5437-486e-9d9b-cb37f52e6af3;mammalia;rodentia;dasyproctidae;dasyprocta;leporina;red-rumped agouti"

FIXTURE_LABELS = [JAGUAR, PUMA, OCELOT, TAPIR, DASYPROCTA_GENUS, AGOUTI]
