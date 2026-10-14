# IZMERNI PROTOKOL — prvi ⊢ rezultat
## Ali režim ⊢/⚠ merljivo zmanjša hibride v izhodu modela v slovenščini?

> Protokol, ne obljuba. Vsak korak izmerljiv; vsak rezultat nosi ⊢ (izmerjen) ali ⚠ (odprt).
> Po L4: poročilo po dejanju, nikoli pred njim. Po L2: izvajalec ne zapečati lastne izmere — rezultat prebere druga roka (recenzent, drug model, revizija).
> Datum protokola: 2026-10-11. Status: ⚠ PRED IZVEDBO — dokler izmera ne pade, je to načrt, ne dokaz.

---

## 0. Raziskovalno vprašanje

**Ali model, ki deluje pod pogodbo ⊢/⚠ (session-scoped lexicon contract), proizvede merljivo manj hibridov in nepreverjenih trditev kot isti model brez pogodbe?**

Ničelna hipoteza (H0): pogodba ne spremeni števila hibridov / nezapečačenih trditev.
Alternativa (H1): pogodba zmanjša hibride in poveča delež ⊢-zapečačenih trditev.

⚠ Odkrito: možno je, da izmera pokaže H0. Tudi to je rezultat in tudi ga objavimo. Režim, ki se boji lastne izmere, ni režim.

---

## 1. Korpus preizkusa (test set)

- **N vzorcev:** 100–200 pozivov v slovenščini, iz treh skupin:
  1. **Definicijski** („Poimenuj in razloži: X") — kjer padecijo inventivni pomeni
  2. **Strokovni** (pravni/medicinski/tehnološki izrazi) — kjer padecijo hibridi
  3. **Sestavljanje** („Sestavi pojem za Y") — kjer se modeli izmišljajo besede (to je tisto kar naj bi rezalo naše)
- Vsak poziv v paru: **isti poziv, dvakrat** — enkrat brez pogodbe (baseline), enkrat s pogodbo v sistemu (contracted).
- Podatki: pozivi se generirajo **pred** izvedbo, zapečačijo z zgoščeno vrednostjo (hash), da se ne morejo prilagoditi rezultatu.

## 2. Pogodba (neodvisna spremenljivka)

- `wordlist-v0.6-demo.json` (10 ⊢/⚠ + 3 znani hibridi) — obstoječi vir, nič novega
- Dodatno: razširjen wordlist iz zaprtega korpusa (⚠ odprto — višje število vnosov poveča moč izmere; min. 50 vnosov priporočeno)

## 3. Meritve (odvisne spremenljivke)

Za vsak izhod modela (baseline in contracted):

| Meritev | Definicija | Orodje |
|---|---|---|
| **Hibridi** | št. besed, ki niso v SSKJ²/Fran in niso označene ⚠ | `vba.mark_claims` + ročna kontrola vzorca |
| **⊢ delež** | št. trditev z ⊢ vnosom / skupno trditev | `vba.mark_claims` |
| **⚠ delež** | št. trditev, ki ostanejo ⚠ | `vba.mark_claims` |
| **Falls** | št. zapečačenih padcev v register | `vba.Register` |

Sekundarna izmera (za ponovljivost): **BLEU/chrF ne merimo** — merimo le, kar režim mehanično šteje. To je meja protokola: izmeri hibride in zapečatke, ne fluentnost.

## 4. Izvedba (koraki)

1. **Priprava pozivov** (izvajalec): 100–200 pozivov, 3 skupine, hash zapečačen
2. **Baseline izvedba**: model brez pogodbe (isto temperaturo, seme zapečačeno)
3. **Contracted izvedba**: isti pozivi, pogodba naložena v sistemsko sporočilo (lexicon kot session-scoped pogodba — isto kot v demo zvezku)
4. **Označevanje**: vsak izhod skozi `vba.mark_claims` — mehanično, brez občutka
5. **Izračun**: deleži hibridov/⊢/⚠ za oba pogoja
6. **Statistika**: najmanjši pošteni test — deleži + 95 % interval (Wilson), ali Δ presega interval. ⚠ Če je N majhen, je izmera šibka — to zapišemo, ne skrijemo
7. **Register**: vsak najden hibrid zapečačen v register (L3 — tudi če izmera pokaže H0, register raste; izmera ne pobriše padcev)
8. **Poročilo**: tabela izhodov + graf + register kot dodatek — **objavljeno po izvedbi** (L4)

## 5. Predregistracija (da izmera ne more lagati)

Pred izvedbo zapečačimo (commit):
- [ ] seznam pozivov (hash)
- [ ] wordlist (verzija + hash)
- [ ] model + verzija + seme/temperatura
- [ ] statistični test + prag
- [ ] datum izvedbe

Vsak odmik od protokola po izvedbi = nova vrstica v registru, ne tiho prepisovanje (L3).

## 6. Odkrite meje (zapisane pred izvedbo, ne po njej)

- ⚠ Pogodba je sistemsko sporočilo: model je *lahko* ignorira — izmera meri, ali ignoriira ali ne, kar je samo po sebi rezultat
- ⚠ Slovar (SSKJ²/Fran) je baza za hibride, ampak **slovar ni svet**: novih besed ni v slovarju in niso hibridi (npr. neologizmi). Zato: ročna kontrola vzorca 20 % izhodov, da izmera ne šteje neologizmov kot hibridov
- ⚠ En model, en jezik: izmera je prvi ⊢, ne zadnji — posploševanje na „vse modele" bi bilo napaka

## 7. Kaj po uspehu (H1) in kaj po neuspehu (H0)

- **H1**: rezultat ⊢ → vključitev v ponovno prijavo na clarinsi (prva številka, ki jo pregled ne more zavrniti s „nejasen opis") + ista številka v e-pošti Mistralu
- **H0**: rezultat ⊢ → register padcev raste, protokol se popravi (več vnosov, druga skupina pozivov), izvedba se ponovi z razliko (stotnija 107). Objavimo tudi H0 — režim, ki objavljuje lastne padce, je edini, ki mu lahko verjamemo

---

*Protokol zapečačen pred izvedbo. Izvedba = izvajalčeva roka (pozivi, dostop do modela, slovensko branje izhodov). Poročilo = po dejanju.*

---

## 8. Posodobitev protokola (2026-10-11, po izmeri okolja — register vnos 5)

**Izmera okolja (⊢, izvedena):** živi tek iz sandboxa trenutno ni mogoč —
MISTRAL_API_KEY odsoten; HF žeton odsoten; povejmo.si brez javnega API-ja;
5 GB RAM / brez GPU ne nese GaMS-9B (~18 GB BF16).

**Tri odprte poti do živega teka (vse pripravljene, natanko ena spremenljivka na vsako):**

| Pot | Kaj je potrebno | Kdor drži ključ |
|---|---|---|
| (a) Mistral API | `MISTRAL_API_KEY` v okolju tekača | izvajalec (vnos v env, nikoli v repozitorij) |
| (b) HuggingFace inference | HF žeton s ponudnikom inferenc za `cjvt/GaMS-9B-Instruct` (model odprt, ne-gated ⊢ izmerjeno 2026-10-11) | izvajalec |
| (c) GPU gostitelj | isti tekač, `--live`, na stroju z ≥24 GB VRAM; uteži odprte | izvajalec ali nacionalna infrastruktura (PoVeJMo/ARIS) |

**Pot (b) je prava za misijo:** GaMS je nacionalni model — izmera pogodbe
⊢/⚠ **na GaMS** nosi največjo težo za slovenski pregled (clarinsi) in za
nacionalni program: rezultat pove, ali nacionalni model spoštuje lastno
slovenščino pod pogodbo. Mistral pot (a) ostane kot kontrolna skupina
(tuji model, isti pozivi) — dve roki po L2: dva modela, ena izmera.

** Razširitev protokola (ista izmera, dva modela):** če sta oba ključa
dosegljiva, teče isti tekač dvakrat — GaMS (domači) in Mistral (tuji) —
in poročilo nosi oba stolpca. To je poštena oblika zahteve "Slovenian as
first-class": ne pridiganje, izmera.

**Naslednji korak (osmišljen do ene vrstice):**
```
HF_TOKEN=hf_… MISTRAL_API_KEY=… python3 run_izmera.py --live --model cjvt/GaMS-9B-Instruct
```
(s preklopom ponudnika v `call_model` — endpoint se nastavi z istim
argumentom; tekač je pripravljen, potrjen s dry-run.)

**Zaporedje po živem teku:**
1. rezultat ⊢ → `rezultat-v1.0.md` se prepiše z živimi številkami (ne
   canned) in nosi žig "IZMERNI REZULTAT — živ tek"
2. register se dopolni z novimi hibridi, najdenimi v živem izhodu (L3)
3. ponovna prijava na clarinsi: v1.1 README-SL (izvajalčeva roka) +
   delujoča koda + prva številka — vsaka pripomba recenzije odgovorjena z
   izmerjenim telesom
4. ista številka gre v e-pošto Mistralu (ponovitev z razliko, stotnija 107)

## 9. Zapis o CI izvedbi (2026-10-11, Z4 — po dejanju)

Push `ff6d954` je sprozil prvi CI tek. Izid: failure — dve izmeri:
1. stara pokvarjena YAML je se registrirana kot "invalid workflow" na
   GitHubovi strani (name se bere kot pot); teka je padel se pred koraki
2. App ne more sproziti workflow_dispatch (403, izmerjeno) — zato push
   trigger: vsak push na meritvene datoteke zene teka sam, brez gumba

Push tega odseka cisti zastarelo stanje: veljavna YAML + nov push =
registriran teka z imenom "Izmera zivi tek". Naslednji tek bo prvi
polni CI tek (dry-run korak, potem zivi ce zeton v Secrets obstaja).
