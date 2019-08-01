# Autor

- Peter Gazdík, xgazdi03@stud.fit.vutbr.cz

# Závislosti

Okrem bežnej inštalácie sú pre správne fungovanie skriptov potrebné nasledovné nástroje:

- Python balíčky: `numpy, keras, tensorflow, numpy_indexed, kaldi_io, matplotlib, scikit_learn, matplotlib2tikz`.
- Nástroj `SRILM` pre vytváranie N-gramových jazykových modelov.
- Utilitu `flac` pre prípravu dát v datasetoch foxforge.
- Sequitor G2P pre prevod slov na fonémy, ktoré nie sú v slovníku.

# Adresárová štruktúra

- `conf/` - Konfiguračné súbory s parametrami pre jednotlivé datasety.
- `local/data` - Skripty zabezpečujúce prípravu dát.
- `local/pron` - Zdrojové súbory v jazyku Python, pomocou ktorých sú implementované všetky metódy vyhodnotenia výslovnosti.
- `local/steps` - Shell skripty zabezpečujúce trénovanie DNN-HMM akustických modelov, klasifikátorov výslovnsti a evaluáciu jednotlivých experimentov.
- `local/utils` - Podporné nástroje pre vyššie uvedené skripty a vyhodnocovanie výsledkov.

# Spustenie experimentov

Spúšťanie experimentov je možné dvoma spôsobmi, pomocou skriptu `run.sh` alebo `experiment.sh`. Pred spustením je však nutné nastaviť cestu k datasetom v skripte `run.sh`.

Skript `run.sh` má podobnú štruktúru ako obdobné skripty v Kaldi datasete. Skript vyžaduje jeden povinný parameter s názvom datasetu, nad ktorým bude spúšťaný. Okrem toho prijíma voliteľné parametre s typom akustického modelu (trifónový, monofónový), typom príznakov použitých pre klasifikáciu (HMM vierohodnosti stavov, fonologické rysy). Pre viac informácii viď obsah súboru. 

Nakoľko spúšťanie vyššie uvedeného súboru je pomerne koplikované, pre jednoduchšie spúšťanie experimentov bol vytvorený ešte jeden skript `experiment.sh`. Tento skript spúšťanie základných experimentov, ktoré boli popísané v rámci technickej správy, vrátane multilingválneho trénovania. Príklad spustenia skriptu:

```
./experiment.sh timit 1 2 3 4 timit,isle 1 2 3 4 10 11 isle+timit-isle 

```

Takéto spustenie povedie na natrénovanie DNN-HMM modelov TIMIT datasetu (parametre `timit 1 2 3 4`. V ďalšom kroku dôjde k trénovaniu multilingválneho DNN-HMM AM nad datasetmi timit a isle (parametre `timit,isle 1 2 3 4`). Následne sú nad takýmto modelom natrénovaný klasifikátor nesprávnej výslovnosti a vyhodnotené GOP skóre (parametre `10 11`). Na záver dôjde k premenovaniu zložky s výsledkami z `exp/isle` na `exp/timit-isle`, z ktorej je zrejmé, o aký multilingválny akustický model sa jedná, a zároveň je možné trénovať iné multilingválne jazykové modely. 
