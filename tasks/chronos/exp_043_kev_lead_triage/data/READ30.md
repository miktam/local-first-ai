# exp_043: 30 test leads for Andrei's read

Check each label against the message and LABELS.md. Mark any lead you would label differently.

**W7-33** (ru, whatsapp)

> здравствуйте хочу продать дом, вы работаете с эстепоной? сколько у вас комиссия

intent `sell` · type `villa` · timeline `unspecified` · financing `None` · foreign `None` · qualified `False` · urgency `0` · action `ask_details`

**W5-46** (sv, whatsapp)

> Hej! Solceller till era objekt? Vi installerar på hela kusten och ger 20% provision till mäklare som rekommenderar oss. Svara JA så skickar vi prislista ☀️

intent `spam` · type `unspecified` · timeline `unspecified` · financing `None` · foreign `None` · qualified `False` · urgency `0` · action `discard`

**W3-39** (pl, whatsapp)

> dzien dobry, czy zajmujecie sie tez lokalami uzytkowymi? mam lokal w San Pedro, 80 m2, wynajety pod sklep, chce go sprzedac. jak to u was wyglada

intent `sell` · type `commercial` · timeline `unspecified` · financing `None` · foreign `None` · qualified `False` · urgency `0` · action `ask_details`

**W8-03** (en, voice)

> Hello, hi, yes, I was looking at your, the villa listings, the ones in Benahavís, La Zagaleta and all that, very nice, very nice portfolio. And I was going to ask about buying one but, well, to be honest with you, the real reason I'm calling is, we're a reformas company, we do kitchens, bathrooms, full refurbishments, we've done like forty villas in the area, and I thought, your buyers, when they buy, they always want to change things, so maybe we could have, um, an arrangement, a referral thing, you send us the buyer, we give you a percentage. I can send the catalogue, no problem. Let me know, ok.

intent `spam` · type `None` · timeline `unspecified` · financing `None` · foreign `None` · qualified `False` · urgency `0` · action `discard`

**W2-20** (es, voice)

> Hola, mira, te explico. Nosotros somos venezolanos, vivimos en Caracas, y queremos irnos para España. Lo que queremos es, este, venir primero en diciembre como dos semanas para conocer la zona, o sea un alquiler de vacaciones, pero eso es lo de menos, eso lo resolvemos por nuestra cuenta si hace falta. Lo que de verdad buscamos es un alquiler de un año o más, en Estepona o Manilva, algo de tres habitaciones, puede ser casa o apartamento, nos da igual. Podemos pagar hasta mil ochocientos al mes. La idea es estar instalados antes de que termine el año. ¿Tienen algo así? Gracias.

intent `rent_long` · type `unspecified` · timeline `months` · financing `None` · foreign `True` · qualified `True` · urgency `1` · action `send_listings`

**W8-42** (en, voice)

> Hola, buenas, um, I have an ático, a penthouse, in Marbella centro, near the Alameda, three bedrooms, big terrace, it's a good one. I'm from here, from Málaga, but I'm moving to Madrid for work, so I need to sell it, fast, like, this month if possible, or next. I don't know the price, honestly, prices have gone loco, so I need you to tell me what it's worth and we put it on the market. Can you come this week? Gracias.

intent `sell` · type `penthouse` · timeline `now` · financing `None` · foreign `False` · qualified `True` · urgency `2` · action `book_valuation`

**W4-31** (de, voice)

> ja hallo, ähm, wir wollen unser haus in alhaurín el grande verkaufen, so 550 bis 600 tausend stellen wir uns vor, und dann was kleineres kaufen, ne wohnung näher am meer. aber erstmal muss das haus weg, am besten noch dieses jahr. ähm, könnten sie mal vorbeikommen und sich das anschauen? wir sind eigentlich immer da.

intent `sell` · type `villa` · timeline `months` · financing `None` · foreign `disagree` · qualified `True` · urgency `1` · action `book_valuation`

**W6-12** (fr, whatsapp)

> bonjour on regarde un peu ce qui se fait sur la costa del sol pour acheter un jour, rien de precis pour l instant, ni budget ni endroit. vous avez une brochure ou une newsletter ?

intent `buy` · type `unspecified` · timeline `unspecified` · financing `None` · foreign `None` · qualified `False` · urgency `0` · action `ask_details`

**W6-18** (fr, email)

> Bonjour,

Mon mari est muté à Málaga à partir de janvier et nous cherchons à acheter une maison de 4 chambres entre Benalmádena et Fuengirola, si possible proche d'une école internationale. Notre budget est d'environ 700k, avec un prêt pour une partie (60 % environ). Nous vivons actuellement à Nantes et venons repérer sur place du 7 au 11 novembre.

Pourriez-vous nous préparer une sélection à visiter pendant ces dates ?

Merci,
Céline

intent `buy` · type `villa` · timeline `now` · financing `True` · foreign `True` · qualified `True` · urgency `2` · action `send_listings`

**W3-45** (pl, email)

> Dzień dobry, sprzedajemy nasz penthouse w Estepona (kupiony w 2019 za 410k, 3 sypialnie, solarium) i w zamian chcielibyśmy kupić mniejsze mieszkanie w tej samej okolicy. Najpierw sprzedaż, chcemy to załatwić jeszcze w tym roku. Ile dziś może być wart i czy warto odświeżyć przed wystawieniem? Pozdrawiam, Joanna

intent `sell` · type `penthouse` · timeline `months` · financing `None` · foreign `None` · qualified `True` · urgency `1` · action `book_valuation`

**W3-35** (pl, voice)

> Dzień dobry, mam takie pytanie, yyy, jak to jest teraz z tą Golden Visa w Hiszpanii, bo słyszałem że ją zlikwidowali, a znajomy mówi że jeszcze da się coś zrobić przez nieruchomości… czy to w ogóle dotyczy obywateli Unii? Tak tylko pytam, żeby wiedzieć

intent `info` · type `unspecified` · timeline `unspecified` · financing `None` · foreign `None` · qualified `False` · urgency `0` · action `answer_question`

**W7-21** (ru, email)

> Здравствуйте! Мы семья из Алматы, переезжаем на Коста дель Соль весной 2027 года, дети пойдут в международную школу (рассматриваем Сотогранде или Эстепону). Хотим купить таунхаус с 3–4 спальнями, бюджет до 900 тысяч, часть суммы планируем взять в ипотеку в испанском банке как нерезиденты — подскажете, реально ли это сейчас? Будем в Марбелье 15–20 декабря, хотели бы посмотреть варианты. С уважением, Алия

intent `buy` · type `townhouse` · timeline `months` · financing `True` · foreign `True` · qualified `True` · urgency `1` · action `send_listings`

**W8-41** (en, whatsapp)

> apartment near the beach in fuengirola next weekend fri-mon, 2 adults, anything?

intent `rent_holiday` · type `apartment` · timeline `now` · financing `None` · foreign `None` · qualified `True` · urgency `2` · action `send_listings`

**W5-42** (sv, form)

> Vi bor i Linköping och vill prova på Estepona: två veckor i februari i en lägenhet, och om vi trivs vill vi teckna ett hyreskontrakt på ett år från april. Målet är långtidskontraktet. För långtid kan vi betala max 1.800 €/månad, två sovrum, gärna i Estepona centrum eller vid hamnen.

intent `rent_long` · type `apartment` · timeline `later` · financing `None` · foreign `True` · qualified `True` · urgency `0` · action `send_listings`

**W8-20** (en, voice)

> Hi, um, this is a bit of a random one, we're not buying yet, we're just, um, researching, we're looking at maybe moving in a couple of years and I wanted to ask about Manilva, and Sabinillas, what's it like for families, are there international schools nearby, is it, is it dead in winter, that kind of thing. We're not asking for properties, honestly, just your opinion on the area. Thank you.

intent `info` · type `unspecified` · timeline `later` · financing `None` · foreign `None` · qualified `False` · urgency `0` · action `answer_question`

**W5-35** (sv, voice)

> Hej, eh, jag fick ert nummer av en bekant som sa att ni kan området bra. Vi är en barnfamilj som funderar på kusten på sikt, och jag undrar lite vad skillnaden är mellan San Pedro och Estepona, alltså för familjer, skolor, hur det är på vintern och så. Vi är inte redo att göra nånting än, vi samlar bara information. Tack så mycket.

intent `info` · type `unspecified` · timeline `None` · financing `None` · foreign `None` · qualified `False` · urgency `0` · action `answer_question`

**W4-12** (de, email)

> Guten Tag,

wir besitzen eine 3-Zimmer-Wohnung in Riviera del Sol (Mijas Costa), die ab 1. November frei wird, und möchten sie langfristig vermieten, Mietvorstellung 1.400 €/Monat. Da wir in Düsseldorf leben, suchen wir eine Agentur, die Vermietung und Verwaltung vor Ort übernimmt.

Bitte rufen Sie mich kurz an, um das Weitere zu besprechen: +44 7700 900141.

Mit freundlichen Grüßen
Petra S.

intent `rent_long` · type `apartment` · timeline `now` · financing `None` · foreign `True` · qualified `True` · urgency `2` · action `call_now`

**W1-39** (en, whatsapp)

> hi looking for land with licence to build 10-15 townhouses, estepona/cancelada side, budget 2-3M for the plot. we're a developer from madrid, have done 3 projects on the coast already

intent `invest` · type `plot` · timeline `unspecified` · financing `None` · foreign `False` · qualified `True` · urgency `0` · action `send_listings`

**W6-03** (fr, whatsapp)

> bonjour je cherche un appartement a louer a l annee a fuengirola, 2 chambres, max 1200/mois, a partir de debut decembre. de preference proche de la gare. merci 🙏

intent `rent_long` · type `apartment` · timeline `months` · financing `None` · foreign `None` · qualified `True` · urgency `1` · action `send_listings`

**W8-06** (en, voice)

> Yeah hi, um, quick question really, not looking to buy or sell anything, we're Dutch, we have a place in Mijas, La Cala actually, and um, my brother-in-law said something about the plusvalía, the municipal one, that it changed, or that you can choose how it's calculated now? I just want to understand how it works. We're not selling, not planning to, it's just, I like to know these things. Just how the plusvalía works. Thanks.

intent `info` · type `unspecified` · timeline `unspecified` · financing `None` · foreign `True` · qualified `False` · urgency `0` · action `answer_question`

**W2-16** (es, form)

> Villa con piscina (climatizada si es posible) en Marbella o alrededores para 10 personas, del 20 al 27 de diciembre. Presupuesto hasta 6.000 € la semana.

intent `rent_holiday` · type `villa` · timeline `months` · financing `None` · foreign `None` · qualified `True` · urgency `1` · action `send_listings`

**W5-26** (sv, form)

> Hur lång tid tar det att få NIE-nummer, och behöver man ha det innan man börjar leta bostad eller räcker det att ha det vid notarien? Jag läser på inför framtiden, har inget konkret på gång.

intent `info` · type `unspecified` · timeline `unspecified` · financing `None` · foreign `disagree` · qualified `False` · urgency `0` · action `answer_question`

**W5-36** (sv, whatsapp)

> hej! vi är ett ungt par som vill köpa vår första lägenhet, benalmádena eller torremolinos, 1-2 sov, max 180k. det blir nästa år tidigast, vi måste ta lån. går det att få bolån här? 🙏

intent `buy` · type `apartment` · timeline `later` · financing `True` · foreign `None` · qualified `True` · urgency `0` · action `send_listings`

**W3-04** (pl, form)

> Interesuje mnie oferta nr 4521 (penthouse, Nueva Andalucía). Czy jest nadal dostępna i czy można obejrzeć w tym tygodniu? Przyleciałem z Warszawy, jestem w Marbelli do piątku. Tel. +44 7700 900112.

intent `buy` · type `penthouse` · timeline `now` · financing `None` · foreign `True` · qualified `True` · urgency `3` · action `call_now`

**W5-16** (sv, form)

> Hej, ref. 5512 (takvåningen i Elviria) – är den ledig 20–27 december? Vi är sex personer, tre vuxna och tre barn. Går det att få sen incheckning?

intent `rent_holiday` · type `penthouse` · timeline `months` · financing `None` · foreign `None` · qualified `True` · urgency `2` · action `send_listings`

**W2-44** (es, email)

> Buenos días,

Vivimos en Puerto Banús desde hace diez años y queremos cambiar a una casa más grande, en Benahavís o La Quinta, con cuatro dormitorios y piscina propia. Presupuesto hasta 1,5M. Para ello venderemos nuestro piso actual, pero eso lo haremos después; lo prioritario ahora es encontrar la casa, en los próximos meses.

¿Qué tenéis en esa zona?

Un saludo,
Marta y Luis

intent `buy` · type `villa` · timeline `months` · financing `None` · foreign `False` · qualified `True` · urgency `1` · action `send_listings`

**W6-41** (fr, email)

> Bonjour,

Nous arrivons de Lille à Málaga vendredi prochain pour 10 jours et notre location Airbnb vient d'être annulée. Nous cherchons en urgence un appartement 2 chambres à Fuengirola ou Benalmádena, budget jusqu'à 2 000 € pour le séjour, 3 adultes.

Merci de m'appeler dès que possible au +44 7700 900706.

Cordialement,
Laure

intent `rent_holiday` · type `apartment` · timeline `now` · financing `None` · foreign `True` · qualified `True` · urgency `2` · action `call_now`

**W7-27** (ru, email)

> Добрый день! Ищу виллу на неделю в феврале (примерно 12–19 февраля) на юбилей мужа, 10–12 человек, 5–6 спален, Бенахавис или Нуэва Андалусия, обязательно с видом и подогреваемым бассейном. Бюджет до 8 000 € за неделю. Можно ли с поваром по запросу? Жду варианты. Елена

intent `rent_holiday` · type `villa` · timeline `months` · financing `None` · foreign `None` · qualified `True` · urgency `1` · action `send_listings`

**W7-38** (ru, email)

> Здравствуйте, у нас квартира в Эстепоне, мы не живём в Испании постоянно. Объясните, пожалуйста, что такое налог нерезидента (modelo 210), сколько и когда платить, если квартира не сдаётся? И кто должен его подавать? Заранее спасибо. Павел

intent `info` · type `apartment` · timeline `unspecified` · financing `None` · foreign `True` · qualified `False` · urgency `0` · action `answer_question`

**W3-21** (pl, whatsapp)

> czesc, mieszkam w Maladze od 5 lat, mam NIE i umowe na stale, szukam mieszkania 2 syp do 250k w Teatinos albo centrum, z kredytem (mam wstepna zgode z banku). szukam na juz bo umowa najmu mi sie konczy

intent `buy` · type `apartment` · timeline `now` · financing `True` · foreign `False` · qualified `True` · urgency `2` · action `send_listings`

