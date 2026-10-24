# exp_050: 30 test leads for Andrei to read before signing

*Drawn at random (seed 50:read30) from the test set. Each shows the message, the agreed labels and the homes that fit under the fit rule. Note anything you would label differently.*

## y1-36 · en · standard

> Subject: Commercial unit as an investment
> 
> Hello,
> 
> I'm looking at commercial premises as an investment: a retail or office unit I can let out, in Marbella centre or San Pedro. Budget up to 700k. Timing is Q1 2027; I'm not in a rush and would rather wait for the right unit.
> 
> Happy to talk it through on +44 7700 900 512 (UK mobile; I split my time between London and Boston).
> 
> Best,
> Michael

- **invest** · commercial · timeline later · qualified True · urgency 0 · **send_listings**
- budget 700000 · areas ['marbella_centre', 'san_pedro'] · bedrooms None · features –
- fits: 2 — EX-101 (commercial, San Pedro de Alcántara, 320,000), EX-109 (commercial, Marbella centre, 640,000)

## y6-20 · sv · standard

> Hejsan! Jag o min sambo letar lägenhet i Nueva Andalucia, 2 sovrum o terrass e ett måste, max 600k. Vi tar bolån i Spanien (har pratat med banken redan). Vill gärna ha köpet klart innan nyår. Nära golfen vore ju fint men inte nödvändigt 😊 har ni nåt? Mvh Elin

- **buy** · apartment · timeline months · qualified True · urgency 1 · **send_listings**
- budget 600000 · areas ['nueva_andalucia'] · bedrooms 2 · features ['terrace']
- fits: 1 — EX-116 (apartment, Nueva Andalucía, 600,000)

## y1-23 · en · question

> quick question, do i need a tourist licence to rent out my apartment in estepona on airbnb for a few weeks a year? and can the community block it? not sure where to start

- **info** · apartment · timeline unspecified · qualified False · urgency 0 · **answer_question**
- budget None · areas – · bedrooms None · features –
- fits: 0

## y6-44 · sv · standard

> Hej!
> 
> Vi är nere i Marbella över helgen (åker hem söndag kväll) och skulle vilja se villor i Elviria imorgon lördag eller på söndag förmiddag, om det går att ordna med så kort varsel.
> 
> Vi letar en villa med minst 3 sovrum, havsutsikt och gångavstånd till stranden, båda är krav. Budget upp till 1,8 miljoner euro, vi har lånelöfte från vår svenska bank på en del av beloppet. Vi har tittat på två hus med en annan mäklare i Elviria idag men de var inte rätt, så vi vill gärna se fler innan vi bestämmer oss.
> 
> Ring mig: +44 7700 900 129, jag svarar hela dagen.
> 
> Mvh
> Helena Strand

- **buy** · villa · timeline now · qualified True · urgency 3 · **call_now**
- budget 1800000 · areas ['elviria'] · bedrooms 3 · features ['sea_view', 'walk_to_beach']
- fits: 2 — EX-134 (villa, Elviria, 1,960,000), EX-151 (villa, Elviria, 1,770,000)

## y2-01 · en · standard

> hi there, me and my wife looking for apartment in nueva andalucia, 2 bedroom minimum, max 650k. we fly in 2nd week of november to see some, can u call me on 07700 900 123? terrace wud be nice but not a must. we live in utrecht so weekend viewings is best. groeten pieter

- **buy** · apartment · timeline now · qualified True · urgency 2 · **call_now**
- budget 650000 · areas ['nueva_andalucia'] · bedrooms 2 · features –
- fits: 2 — EX-116 (apartment, Nueva Andalucía, 600,000), EX-131 (apartment, Nueva Andalucía, 710,000)

## y2-37 · en · standard

> Name: Suresh. Message: I am NRI based in Dubai, looking to buy commercial premises (shop or office) in Marbella centre or San Pedro to rent out for rental income. Budget 700k euro, cash, funds are ready. Want to close as soon as possible, ideally within this month. Call me +44 7700 900 321 any time.

- **invest** · commercial · timeline now · qualified True · urgency 2 · **call_now**
- budget 700000 · areas ['marbella_centre', 'san_pedro'] · bedrooms None · features –
- fits: 2 — EX-101 (commercial, San Pedro de Alcántara, 320,000), EX-109 (commercial, Marbella centre, 640,000)

## y1-43 · en · standard

> Subject: Renting for a year before we buy
> 
> Dear team,
> 
> My husband and I are retiring to Spain and want to rent for a year before we commit to buying. We're looking for a three-bedroom apartment in Elviria or Estepona with a sea view, which is the one thing we insist on, from the middle of January. We can pay up to 3,000 a month. We'd also like a pool, but we could live without one.
> 
> We can come over for viewings whenever suits once you have something to show us.
> 
> Best wishes,
> Margaret and Tony

- **rent_long** · apartment · timeline later · qualified True · urgency 0 · **send_listings**
- budget 3000 · areas ['elviria', 'estepona'] · bedrooms 3 · features ['sea_view']
- fits: 2 — EX-108 (apartment, Estepona, 2,150), EX-138 (apartment, Elviria, 2,800)

## y8-19 · ru · standard

> Да, здравствуйте, это голосовое, извините, за рулём. Значит, мне нужна вилла в аренду, долгосрочно, минимум на год, в Пуэрто Банусе. Три спальни минимум, бассейн обязательно, э-э, и парковка обязательно, у нас две машины. Бюджет до шести тысяч в месяц. Мы уже в Марбелье, в отеле сидим, хотели бы завтра уже что-то посмотреть. Перезвоните мне, пожалуйста, номер плюс сорок четыре, семь семь ноль ноль, девять ноль ноль, два один четыре. Спасибо.

- **rent_long** · villa · timeline now · qualified True · urgency 3 · **call_now**
- budget 6000 · areas ['puerto_banus'] · bedrooms 3 · features ['pool', 'parking']
- fits: 1 — EX-143 (villa, Puerto Banús, 5,900)

## y1-35 · en · no_match

> Name: Fiona
> Message: Hi, we're looking for a 2 bedroom apartment in Marbella old town for 6 nights from 22 December, for Christmas. Budget around 2k for the stay. Do you have anything? Thanks

- **rent_holiday** · apartment · timeline months · qualified True · urgency 1 · **send_listings**
- budget 2000 · areas ['marbella_centre'] · bedrooms 2 · features –
- fits: 0

## y8-08 · ru · question

> Добрый день,
> 
> Я гражданин России, в Испании не резидент. Пока только изучаю вопрос и хочу заранее понять расходы при покупке недвижимости в Андалусии. Подскажите, пожалуйста:
> 1. Какой налог платится при покупке вторички и сколько это в процентах?
> 2. Нужен ли NIE до подписания договора резерва и как долго его делают?
> 3. Какие ещё расходы кроме налога: нотариус, регистрация, юрист?
> 
> Заранее спасибо.
> Дмитрий, dmitry.k@example.com

- **info** · unspecified · timeline unspecified · qualified False · urgency 0 · **answer_question**
- budget None · areas – · bedrooms None · features –
- fits: 0

## y3-23 · pl · question

> czesc, mam pytanie - czy w Marbelli trzeba miec licencje turystyczna zeby wynajmowac mieszkanie turystom na krotko? i czy wspolnota moze tego zabronic? dzieki

- **info** · apartment · timeline unspecified · qualified False · urgency 0 · **answer_question**
- budget None · areas – · bedrooms None · features –
- fits: 0

## y8-31 · ru · standard

> Добрый день, мы relocation-агентство, подбираем жильё для сотрудника нашего клиента (IT компания), он переезжает из Тель-Авива с женой. Нужна квартира в Сан Педро, 2 спальни, долгосрочно от года, до 2200 евро в месяц, заезд с 1 ноября. Договор будет на компанию. Звоните 07700 900 321, Яна

- **rent_long** · apartment · timeline now · qualified True · urgency 2 · **call_now**
- budget 2200 · areas ['san_pedro'] · bedrooms 2 · features –
- fits: 1 — EX-149 (apartment, San Pedro de Alcántara, 2,050)

## y6-28 · sv · no_match

> hej! har ni nån lägenhet i puerto banus att hyra v. 8 2027 (sportlovet)? vi e 4 pers, max 1500 för veckan. /Jocke

- **rent_holiday** · apartment · timeline later · qualified True · urgency 0 · **send_listings**
- budget 1500 · areas ['puerto_banus'] · bedrooms None · features –
- fits: 0

## y8-10 · ru · spam

> Уважаемый пользователь Idealista,
> 
> В ваш профессиональный аккаунт был выполнен вход с нового устройства. Если это были не вы, срочно подтвердите владение аккаунтом, иначе все ваши объявления будут сняты с публикации в течение 24 часов.
> 
> Для подтверждения ответьте на это письмо, указав логин и пароль, или напишите в службу безопасности: seguridad-idealista@example.com
> 
> Служба безопасности Idealista

- **spam** · unspecified · timeline unspecified · qualified False · urgency 0 · **review**
- budget None · areas – · bedrooms None · features –
- fits: 0

## y4-42 · es · standard

> Buenas tardes. Vivo en Bogotá y quiero comprar un apartamento pequeño, de una habitación, como segunda vivienda para las vacaciones. Lo importante es que quede a distancia caminable de la playa, en San Pedro o Estepona. Mi presupuesto es de 400.000 euros. No tengo afán, lo estoy mirando para el año entrante. Quedo atento a lo que tengan. Muchas gracias.

- **buy** · apartment · timeline later · qualified True · urgency 0 · **send_listings**
- budget 400000 · areas ['san_pedro', 'estepona'] · bedrooms 1 · features ['walk_to_beach']
- fits: 1 — EX-106 (apartment, San Pedro de Alcántara, 380,000)

## y8-32 · ru · no_match

> Имя: Руслан
> Сообщение: Здравствуйте, ищем квартиру в Сотогранде, 3 спальни, бюджет до 700 тысяч евро. Только Сотогранде, нам нужно рядом с поло-клубом. Покупка до конца года. Спасибо.

- **buy** · apartment · timeline months · qualified True · urgency 1 · **send_listings**
- budget 700000 · areas ['other'] · bedrooms 3 · features –
- fits: 0

## y6-37 · sv · standard

> Hej! Jag skriver för mina föräldrar som är 70+ och inte så bra på att mejla 🙂 De vill köpa en lägenhet i San Pedro eller Guadalmina, 2 sovrum, garageplats är ett måste (pappa vill inte leta parkering), gärna hiss. Max 750k, de betalar kontant från husförsäljningen i Sverige. De kommer ner på måndag och är här i två veckor så det vore toppen om ni kan visa några lägenheter då. Ring mig på 07700 900 918 så samordnar jag. /Karin

- **buy** · apartment · timeline now · qualified True · urgency 2 · **call_now**
- budget 750000 · areas ['san_pedro', 'guadalmina'] · bedrooms 2 · features ['parking']
- fits: 1 — EX-140 (apartment, Guadalmina, 770,000)

## y7-45 · fr · standard

> Bonjour, je dispose d'environ 400 000 € pour un achat, qu'est-ce que vous me conseilleriez ? Merci

- **buy** · unspecified · timeline unspecified · qualified False · urgency 0 · **ask_details**
- budget 400000 · areas – · bedrooms None · features –
- fits: 1 — EX-106 (apartment, San Pedro de Alcántara, 380,000)

## y3-28 · pl · standard

> Dzień dobry,
> 
> w pierwszym kwartale 2027 przeprowadzamy się z Wrocławia do Hiszpanii na stałe, mąż dostał kontrakt w Marbelli. Szukamy domu szeregowego do kupienia w San Pedro albo w Guadalminie, minimum 3 sypialnie. Mamy dwójkę dzieci, więc bliskość szkoły to warunek konieczny. Basen byłby plusem, ale nie jest niezbędny. Budżet do 950 tys. euro, będziemy potrzebować kredytu hipotecznego na część kwoty. Czy mogą Państwo przesłać oferty?
> 
> Pozdrawiam,
> Magdalena

- **buy** · townhouse · timeline later · qualified True · urgency 0 · **send_listings**
- budget 950000 · areas ['san_pedro', 'guadalmina'] · bedrooms 3 · features ['near_schools']
- fits: 1 — EX-118 (townhouse, San Pedro de Alcántara, 910,000)

## y1-09 · en · standard

> Subject: Buy-to-let apartments
> 
> Hi,
> 
> I'm based in Dublin and building a small rental portfolio on the coast. I want to pick up two or three apartments for long-term letting in Estepona or San Pedro, up to 550k each. This is about yield, not lifestyle, so I'm not fussy about views. Parking is a plus but not essential.
> 
> I'd like to complete on at least one before the end of the year. I'll probably put a mortgage on one and pay cash for the rest.
> 
> What do you have?
> 
> Regards,
> Conor

- **invest** · apartment · timeline months · qualified True · urgency 1 · **send_listings**
- budget 550000 · areas ['estepona', 'san_pedro'] · bedrooms None · features –
- fits: 3 — EX-106 (apartment, San Pedro de Alcántara, 380,000), EX-146 (apartment, San Pedro de Alcántara, 470,000), EX-154 (apartment, Estepona, 550,000)

## y2-30 · en · no_match

> Hello, we are group of 6 friends from Mumbai coming to Marbella for a wedding, we need an apartment (3 bedrooms) in Puerto Banus for 10 nights from 20 November to 30 November. Budget for the whole stay around 3000 euro. Pool in the building would be nice. Please call me on +44 7700 900 222 (UK sim) as we want to confirm quickly. Thanks, Priya

- **rent_holiday** · apartment · timeline now · qualified True · urgency 2 · **call_now**
- budget 3000 · areas ['puerto_banus'] · bedrooms 3 · features –
- fits: 0

## y3-27 · pl · standard

> Dzien dobry, pisze w imieniu rodzicow, nie znaja hiszpanskiego ani angielskiego, przeprowadzaja sie z Polski na stale. Szukaja mieszkania do wynajecia na dlugo, min rok, w Estepona. 3 sypialnie bo bedziemy ich odwiedzac z dziecmi, budzet do 2300 miesiecznie, od grudnia. Prosze pisac do mnie, moj numer 07700900987

- **rent_long** · apartment · timeline months · qualified True · urgency 1 · **send_listings**
- budget 2300 · areas ['estepona'] · bedrooms 3 · features –
- fits: 1 — EX-108 (apartment, Estepona, 2,150)

## y4-38 · es · no_match

> Hola, somos una pareja de Madrid y buscamos un apartamento en Estepona para la semana de Navidad, del 22 al 29 de diciembre, para dos personas. Presupuesto unos 1.200 € la semana. Mi teléfono es +44 7700 900 644. Gracias.

- **rent_holiday** · apartment · timeline months · qualified True · urgency 1 · **send_listings**
- budget 1200 · areas ['estepona'] · bedrooms None · features –
- fits: 0

## y1-37 · en · spam

> Hi! Im Marta from Sunline Properties in Sotogrande, we're opening an office in Marbella in January and looking for agencies to share listings with, 50/50 on commission. We've got a strong Scandinavian buyer base. Fancy a coffee next week to see if we can work together?

- **spam** · unspecified · timeline unspecified · qualified False · urgency 0 · **review**
- budget None · areas – · bedrooms None · features –
- fits: 0

## y4-34 · es · spam

> Estimado señor/señora,
> 
> Me llamo Abg. Henry Mendoza y represento a un cliente venezolano, alto funcionario, que necesita trasladar con urgencia la suma de 4.500.000 USD fuera del país mediante la compra de propiedades en Marbella. Por motivos de confidencialidad no puede operar directamente. Si acepta recibir los fondos en su cuenta de agencia para la compra, le corresponderá una comisión del 10%. Solo necesitamos que adelante los gastos de legalización bancaria (2.900 EUR) para liberar la transferencia. Responda con sus datos bancarios y copia de su documento de identidad.
> 
> Atentamente,
> Abg. Henry Mendoza

- **spam** · unspecified · timeline unspecified · qualified False · urgency 0 · **review**
- budget None · areas – · bedrooms None · features –
- fits: 0

## y2-03 · en · standard

> Dobryi den. I am moving from Kyiv for job in Marbella (IT company). need apartment long term, minimum 1 year, 2 bedrooms or more, in nueva andalucia or san pedro, budget up to 2500 per month. start 1 december. garage would be good but ok without. thank u, Oleksandr

- **rent_long** · apartment · timeline months · qualified True · urgency 1 · **send_listings**
- budget 2500 · areas ['nueva_andalucia', 'san_pedro'] · bedrooms 2 · features –
- fits: 1 — EX-149 (apartment, San Pedro de Alcántara, 2,050)

## y3-38 · pl · no_match

> Dzień dobry,
> 
> szukamy willi na tydzień w okresie świąt, od 22 do 29 grudnia, w Marbelli, dla 6 osób (dwie rodziny). Koniecznie z basenem, podgrzewany byłby ideałem. Budżet ok. 4000 euro za tydzień. Czy mają Państwo coś wolnego?
> 
> Pozdrawiam,
> Bartek

- **rent_holiday** · villa · timeline months · qualified True · urgency 1 · **send_listings**
- budget 4000 · areas ['san_pedro', 'nueva_andalucia', 'guadalmina', 'marbella_centre', 'puerto_banus', 'elviria'] · bedrooms None · features ['pool']
- fits: 0

## y1-42 · en · no_match

> Hi 👋 first time buyers here, looking for a 2 bed apartment in Puerto Banus or Nueva Andalucia, max 400.000 €, we have a mortgage agreed in principle. Would like to buy before the end of the year. What's available? Can call me on 07700 900 733

- **buy** · apartment · timeline months · qualified True · urgency 1 · **send_listings**
- budget 400000 · areas ['puerto_banus', 'nueva_andalucia'] · bedrooms 2 · features –
- fits: 0

## y1-18 · en · spam

> Subject: CONFIDENTIAL PROPERTY INVESTMENT PROPOSAL
> 
> Dear Esteemed Agent,
> 
> I am Mrs. Grace, widow of the late Chief Executive of a mining company in West Africa. Before his death my husband deposited a very substantial sum with a security company in Europe, which I now wish to invest in luxury properties in Marbella through a trusted agent such as yourself.
> 
> I will offer you 10% of the total value for your assistance. To release the funds the security company requires a demurrage and clearance fee of 2,850 EUR which I cannot pay from my present location. Kindly send this amount by Western Union and I will forward the release documents within 48 hours.
> 
> Please treat this with utmost confidentiality.
> 
> Yours sincerely,
> Mrs. Grace

- **spam** · unspecified · timeline unspecified · qualified False · urgency 0 · **review**
- budget None · areas – · bedrooms None · features –
- fits: 0

## y3-03 · pl · standard

> czesc szukam mieszkania do wynajecia na rok w estepona 1 sypialnia max 1400 miesiecznie od zaraz bo zaczynam prace 2 listopada. moj nr +44 7700 900 321 mozna dzwonic

- **rent_long** · apartment · timeline now · qualified True · urgency 2 · **call_now**
- budget 1400 · areas ['estepona'] · bedrooms 1 · features –
- fits: 1 — EX-136 (apartment, Estepona, 1,300)
