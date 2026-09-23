# Zadanie č.1 - Parametrizácia genetického algoritmu - Diskusia výsledkov

**Nastavenie:** GA nad knižnicou `genetic_all` (STU FEI, port do Pythonu), populácia 50
jedincov, 10 premenných, 250 generácií, každá konfigurácia spustená 6-krát (rôzne
random seedy) na Schwefelovej funkcii (`ga.schwefel`, doména `[-500,500]`, globálne
optimum **-4189.83**). Elitizmus (`selsort`) prenáša najlepších jedincov bez zmeny do
ďalšej generácie; zvyšok populácie vzniká výberom → krížením (`crossov`, 1-bodové) →
mutáciami (`mutx` = globálna, `muta` = lokálna/aditívna).

Parametrizácia dvoch skúmaných vlastností:

| | Selektívny tlak | Diverzita |
|---|---|---|
| **veľký** | turnajový výber (`seltourn`) + elitizmus 3 jedincov | `mutx_rate=0.15`, `muta_rate=0.25`, `amp=60` |
| **malý** | náhodný výber (`selrand`) + bez elitizmu | `mutx_rate=0.01`, `muta_rate=0.02`, `amp=5` |

## 1. Experimenty a-d (kombinácie tlak × diverzita)

| variant | popis | priemerný best fitness (250. gen.) |
|---|---|---|
| a | veľký tlak + veľká diverzita | -4010.5 |
| b | veľký tlak + malá diverzita | **-4189.8** (≈ optimum) |
| c | malý tlak + veľká diverzita | -2285.3 |
| d | malý tlak + malá diverzita | -2129.4 |

Najvýraznejší rozdiel spôsobuje **selektívny tlak**, nie diverzita. Pri turnajovom výbere
s elitizmom (a, b) GA rýchlo konverguje a spoľahlivo sa dostáva blízko globálneho minima
už okolo 100.-150. generácie. Pri náhodnom výbere (c, d) sa fitness zlepšuje len
minimálne - bez tlaku na lepších jedincov sa GA správa prakticky ako náhodné prehľadávanie
s lokálnymi mutáciami, populácia nekonverguje k žiadnemu konkrétnemu regiónu priestoru.

V rámci "veľký tlak" dvojice dala **malá diverzita (b)** mierne lepší a hlavne stabilnejší
výsledok ako veľká diverzita (a) - silné mutácie totiž pri už takmer optimálnej populácii
zbytočne "rozhadzujú" dobré riešenia späť do priestoru, čo mierne spomaľuje jemné
doladenie. Pri "malý tlak" dvojici je efekt diverzity zanedbateľný, pretože bez
selekčného tlaku sa dobré gény aj tak nedokážu presadiť v populácii.

## 2. Variant e - kompromis

Zvolená konfigurácia: turnajový výber + elitizmus 1 jedinca (miernejší tlak ako b), stredné
hodnoty mutácií (`mutx_rate=0.05`, `muta_rate=0.08`, `amp=20`). Výsledok **-4187.3**,
prakticky rovnako dobrý ako extrémny variant b, ale s o niečo väčšou rozmanitosťou behov
počas prehľadávania (menšie riziko predčasnej konvergencie do lokálneho minima pri
zložitejších/menej "hladkých" úlohách, než je samotná Schwefelova funkcia).

## 3. Experimenty f-i (ablácia operátorov na variante e)

| variant | zmena oproti e | priemerný best fitness |
|---|---|---|
| e | (referencia) | -4187.3 |
| f | bez kríženia | -4185.5 |
| g | bez globálnej mutácie (`mutx`) | -3722.0 |
| h | bez lokálnej mutácie (`muta`) | -4178.0 |
| i | bez oboch mutácií | -3009.1 |

- **Kríženie (f)** má na tomto probléme prekvapivo malý vplyv - výsledok je takmer
  identický s e. Pri reálnej separovateľnej funkcii ako Schwefel (súčet nezávislých
  príspevkov po premenných) mutácie samotné dokážu efektívne prehľadávať priestor gén
  po géne, kríženie hlavne urýchľuje šírenie už nájdených dobrých hodnôt medzi jedincami.
- **Bez globálnej mutácie (g)** výkon citeľne klesá (-3722) a krivky sa "zaseknú" okolo
  50. generácie - bez veľkých náhodných skokov GA nedokáže uniknúť z lokálnych miním
  Schwefelovej funkcie (má ich veľmi veľa).
- **Bez lokálnej mutácie (h)** je pokles minimálny (-4178) - jemné doladenie čiastočne
  nahrádza kríženie a zvyšný priestor prehľadáva globálna mutácia.
- **Bez oboch mutácií (i)** je najhorší výsledok spomedzi f-i (-3009) - populácia rýchlo
  stratí diverzitu (ostáva iba elitizmus + kríženie existujúcich génov) a "zamrzne"
  výrazne ďaleko od optima, krivky sa sploštia už okolo 20. generácie.

**Záver:** na tejto úlohe je z dvojice mutácií kľúčová **globálna mutácia (mutx)** -
zabezpečuje diverzitu potrebnú na únik z lokálnych miním. Kríženie zohráva vedľajšiu
úlohu, lokálna mutácia len jemne dolaďuje.

## 4. Bonus - Eggholder-10 (porovnanie so Schwefelom)

Rovnaká sada a-e zopakovaná na 10-rozmernej Eggholder funkcii (doména `[-512,512]`).

| variant | Schwefel-10 | Eggholder-10 |
|---|---|---|
| a | -4010.5 | -6508.0 |
| b | -4189.8 | -6593.4 |
| c | -2285.3 | -3630.7 |
| d | -2129.4 | -3174.1 |
| e | -4187.3 | **-6957.1** |

Absolútne hodnoty fitness nie sú medzi funkciami priamo porovnateľné (iná škála/tvar
funkcie), ale **relatívny vzorec je rovnaký**: konfigurácie s veľkým selektívnym tlakom
(a, b, e) výrazne prekonávajú konfigurácie s malým tlakom (c, d), a to na oboch
funkciách. Na Eggholder funkcii navyše kompromisný variant e dokonca mierne prekonal
extrémny variant b - Eggholder má členitejší, menej separovateľný reliéf ako Schwefel,
takže o niečo väčšia diverzita (miernejšie mutácie a slabší elitizmus v e oproti b) tu
pomáha unikať z lokálnych miním o niečo lepšie než čisto elitárna stratégia b.

## Zhrnutie

1. Selektívny tlak má na rýchlosť a kvalitu konvergencie väčší vplyv než diverzita.
2. Príliš malý tlak (náhodný výber) GA prakticky degraduje na náhodné prehľadávanie.
3. Rozumný kompromis (e) dosahuje výsledky porovnateľné s extrémne elitárnou
   konfiguráciou (b), a je robustnejší naprieč rôznymi (aj členitejšími) účelovými
   funkciami - pozri Eggholder.
4. Z operátorov je pre túto triedu úloh najdôležitejšia globálna mutácia (diverzita),
   kríženie má druhoradý význam.
