# exp_049: 30 test leads for Andrei's read

Check each label against the message and LABELS.md (today is 21 October 2026). Mark any lead you would label differently.

**Y5-54** (de, email, standard)

> Hallo,

wir ziehen aus Düsseldorf nach Marbella (mein Mann arbeitet ab Dezember remote für seine Firma) und suchen eine Villa zur Langzeitmiete in Benahavís oder Nueva Andalucía, 4 Schlafzimmer, Garten für den Hund, möglichst mit Pool. Mietbeginn 1. Dezember, Laufzeit 2 Jahre, bis 6.000 EUR monatlich. Zwei Kinder (4 und 7).

Was können Sie uns anbieten?

Viele Grüße
Nadine Köhler

intent `rent_long` · type `villa` · timeline `months` · financing `None` · foreign `True` · on behalf `False` · qualified `True` · urgency `1` · action `send_listings` · budget `6000`

**Y7-60** (fr, email, standard)

> Bonjour,

Notre assureur nous demande une estimation à jour de notre villa de Mijas (La Sierrezuela) pour renouveler le contrat, avant la fin novembre. Il ne s'agit pas d'une vente.

Pouvez-vous faire ce type d'estimation et à quelles conditions ?

Merci,
Famille Rey

intent `valuation` · type `villa` · timeline `None` · financing `None` · foreign `None` · on behalf `False` · qualified `True` · urgency `None` · action `book_valuation` · budget `None`

**Y8-12** (ru, voice, standard)

> Алло, здравствуйте, э-э, я инвестор, у меня, значит, есть около двух миллионов, хочу купить несколько квартир под сдачу, ну, чтобы туристическая лицензия была, в Фуэнхироле или Бенальмадене, может три-четыре штуки, доходность интересует процентов шесть. Хочу закрыть до конца года. Перезвоните мне, ноль семь семь ноль ноль девятьсот четыреста пятьдесят один, спасибо.

intent `invest` · type `apartment` · timeline `months` · financing `None` · foreign `None` · on behalf `False` · qualified `True` · urgency `1` · action `send_listings` · budget `2000000`

**Y1-57** (en, whatsapp, hard_timing)

> we want to sell our 2 bed flat in fuengirola by halloween, bit of a rush, we already have somewhere to go, 230k would do us, can u come and see it this week?

intent `sell` · type `apartment` · timeline `now` · financing `None` · foreign `None` · on behalf `False` · qualified `True` · urgency `2` · action `book_valuation` · budget `230000`

**Y6-33** (sv, whatsapp, standard)

> hej finns det byggbar mark i manilva eller casares för 4-6 radhus? vi är två byggare baserade i malmö som vill göra ett mindre projekt, budget för marken runt 1M. gärna med detaljplan klar

intent `invest` · type `plot` · timeline `unspecified` · financing `None` · foreign `True` · on behalf `False` · qualified `True` · urgency `0` · action `send_listings` · budget `1000000`

**Y1-34** (en, voice, hard_valuation)

> hi um we're selling our flat in, um, torremolinos, la carihuela, to buy something bigger up the coast, um can you value it first, um we want to get going on it straight away really, um its a two bed, um we were thinking about two seventy, two eighty, um but you tell us, um and ill give you my number, its oh seven seven double oh, nine hundred, oh six two, um thanks

intent `sell` · type `apartment` · timeline `now` · financing `None` · foreign `None` · on behalf `False` · qualified `True` · urgency `2` · action `book_valuation` · budget `280000`

**Y5-02** (de, email, hard_valuation)

> Sehr geehrte Damen und Herren,

unsere Mutter ist im August verstorben und hat uns drei Geschwistern ihre Wohnung in Fuengirola (Los Boliches, 2 Schlafzimmer, Meerblick) hinterlassen. Für die Erbschaftsteuer und die Aufteilung unter uns brauchen wir eine belastbare Wertermittlung. Ob wir die Wohnung verkaufen oder behalten, haben wir noch nicht entschieden, das hängt auch vom Wert ab. Wir wohnen alle in Deutschland.

Können Sie so eine Bewertung erstellen und was kostet das?

Mit freundlichen Grüßen
Birgit Hoffmann

intent `valuation` · type `apartment` · timeline `unspecified` · financing `None` · foreign `True` · on behalf `False` · qualified `False` · urgency `0` · action `book_valuation` · budget `None`

**Y1-25** (en, email, hard_intermediary)

> Hi,

we're Costa Moves Relocation, we help executives and families moving to the costa del sol (home search, schools, paperwork). we're building a panel of partner agencies and would like to refer our clients to you for purchases and long lets in exchange for 20% of your commission on each deal. we currently have around 15 to 20 families a month.

can we jump on a quick call this week to go over the partnership terms?

best
Ingrid

intent `spam` · type `unspecified` · timeline `unspecified` · financing `None` · foreign `None` · on behalf `False` · qualified `False` · urgency `0` · action `review` · budget `None`

**Y5-59** (de, email, standard)

> Hallo liebes Makler-Team,

wir sind ein Möbelhaus in Mijas mit eigenem Home-Staging-Service. Für Agenturen bieten wir komplette Möbelpakete für leere Objekte (Wohnzimmer, Schlafzimmer, Terrasse) ab 4.900 € oder als Mietstaging ab 390 € pro Monat, Lieferung und Aufbau innerhalb einer Woche. Gestagte Objekte verkaufen sich nachweislich schneller und teurer.

Gerne senden wir Ihnen unseren Katalog. Interesse?

Mit freundlichen Grüßen
Casa Möbel & Staging

intent `spam` · type `unspecified` · timeline `unspecified` · financing `None` · foreign `None` · on behalf `False` · qualified `False` · urgency `0` · action `review` · budget `None`

**Y2-37** (en, form, standard)

> Hi, our townhouse in La Cala (El Lagarejo, 3 bed, 2 bath, roof terrace) is currently listed with another agency but we're not happy, exclusivity ends 31 October and we want to move it to you straight after. We know our price, 495,000, no need to value it again. Can you call me on 07700 900 548 to arrange photos for the first week of November? Thanks, Nicola

intent `sell` · type `townhouse` · timeline `now` · financing `None` · foreign `None` · on behalf `False` · qualified `True` · urgency `3` · action `call_now` · budget `495000`

**Y7-27** (fr, email, standard)

> Bonjour,

Mon épouse et moi (Fribourg, Suisse) cherchons un appartement à louer à Mijas Costa ou La Cala pour dix jours, du 29 octobre au 8 novembre. Budget environ 1.500 € pour le séjour.

Pouvez-vous m'appeler au +44 7700 900 654 pour en parler ?

Merci,
Bernard Aebischer

intent `rent_holiday` · type `apartment` · timeline `now` · financing `None` · foreign `True` · on behalf `False` · qualified `True` · urgency `2` · action `call_now` · budget `1500`

**Y4-48** (es, whatsapp, standard)

> Hola! Nos trasladamos a Marbella por el trabajo de mi marido. La idea es alquilar un piso un año desde el 1 de noviembre (máx 2.000/mes, 3 dorm, zona San Pedro o Nueva Andalucía) y más adelante comprar cuando conozcamos bien la zona, el presupuesto de compra ya lo veremos. Podéis ayudarnos con las dos cosas?

intent `rent_long` · type `apartment` · timeline `now` · financing `None` · foreign `None` · on behalf `False` · qualified `True` · urgency `2` · action `send_listings` · budget `2000`

**Y3-59** (pl, email, standard)

> Dzień dobry,

szukamy domu szeregowego w Cancelada, Atalaya lub Bel Air, 3 sypialnie, z ogródkiem i wspólnym basenem, do 500 000 €. Finansowanie częściowo z kredytu. Chcielibyśmy obejrzeć kilka opcji na początku grudnia, bo wtedy mamy urlop. Mieszkamy obecnie w Madrycie.

Pozdrawiam,
Rafał

intent `buy` · type `townhouse` · timeline `months` · financing `True` · foreign `False` · on behalf `False` · qualified `True` · urgency `1` · action `send_listings` · budget `500000`

**Y8-09** (ru, whatsapp, hard_timing)

> Ищем апартаменты на море на новогодние праздники, заезд 28 декабря, 10 дней, Марбелья или Пуэрто Банус, семья 4 человека, бюджет 3000 евро за все дни

intent `rent_holiday` · type `apartment` · timeline `months` · financing `None` · foreign `None` · on behalf `False` · qualified `True` · urgency `1` · action `send_listings` · budget `3000`

**Y8-67** (ru, whatsapp, standard)

> Здравствуйте, можете сказать примерно, сколько стоит квартира 90 м2 в Лос-Пакосе (Фуэнхирола), 2 спальни, 3 этаж без лифта? Это наша, нужна цифра для страховки, продавать не планируем.

intent `valuation` · type `apartment` · timeline `unspecified` · financing `None` · foreign `None` · on behalf `False` · qualified `False` · urgency `0` · action `book_valuation` · budget `None`

**Y6-10** (sv, form, standard)

> Hej, vi är en familj på fem (barn 6, 9 och 12) som vill hyra en lägenhet i Marbella-området i två veckor i juli, gärna med pool och nära stranden. Budget upp till 3000 € för hela perioden. Vad har ni? Mvh Karin

intent `rent_holiday` · type `apartment` · timeline `later` · financing `None` · foreign `None` · on behalf `False` · qualified `True` · urgency `0` · action `send_listings` · budget `3000`

**Y6-42** (sv, form, standard)

> Hej, vi vill sälja vår tomt i Monte Mayor (Benahavís), 2800 kvm, bygglov sökt. Vad är den värd idag? Vi har inget prisförslag ännu. Mvh Krister

intent `sell` · type `plot` · timeline `unspecified` · financing `None` · foreign `None` · on behalf `False` · qualified `False` · urgency `0` · action `book_valuation` · budget `None`

**Y4-51** (es, email, standard)

> Buenos días,

Necesito una tasación de mi vivienda en Mijas Golf (adosado de 3 dormitorios) para presentarla en el banco: quiero ampliar la hipoteca para reformar la cocina y los baños. No tengo intención de vender.

¿Hacen tasaciones homologadas o solo valoraciones orientativas? ¿Y qué precio tiene el servicio?

Gracias

intent `valuation` · type `townhouse` · timeline `unspecified` · financing `True` · foreign `None` · on behalf `False` · qualified `False` · urgency `0` · action `book_valuation` · budget `None`

**Y3-64** (pl, form, standard)

> Interesuje mnie mieszkanie ref. 7712 w Benalmádenie (Arroyo de la Miel). Czy jest jeszcze dostępne i czy można umówić oglądanie w przyszłym tygodniu? Budżet do 280 000 €, będziemy brać kredyt w Hiszpanii. Kontakt: k.wojcik@example.com

intent `buy` · type `apartment` · timeline `now` · financing `True` · foreign `None` · on behalf `False` · qualified `True` · urgency `2` · action `send_listings` · budget `280000`

**Y8-61** (ru, whatsapp, standard)

> здравствуйте, а что такое комьюнити и сколько обычно платят в месяц в урбанизации с бассейном? и кто отвечает за ремонт крыши, владельцы или управляющая компания

intent `info` · type `unspecified` · timeline `unspecified` · financing `None` · foreign `None` · on behalf `False` · qualified `False` · urgency `0` · action `answer_question` · budget `None`

**Y5-35** (de, email, hard_timing)

> Guten Tag,

wir möchten unsere Wohnung in Torremolinos (La Carihuela, 2 Schlafzimmer, 75 m², Meerblick) im Dezember auf den Markt bringen, sobald die jetzigen Mieter ausgezogen sind. Unsere Preisvorstellung liegt bei 285.000 €. Was benötigen Sie von uns für den Auftrag und wie ist Ihre Provision?

Mit freundlichen Grüßen
Silke und Ralf Neumann

intent `sell` · type `apartment` · timeline `months` · financing `None` · foreign `None` · on behalf `False` · qualified `True` · urgency `1` · action `book_valuation` · budget `285000`

**Y5-13** (de, email, hard_intermediary)

> Hallo liebes Team,

ich bin Immobilienfotografin in Marbella (Innen, Drohne, Twilight-Shootings, Video-Rundgänge). Für Agenturen biete ich ein Komplettpaket ab 180 € pro Objekt an, Lieferung innerhalb von 48 Stunden, bei mehr als 5 Objekten im Monat 15 % Rabatt. Portfolio schicke ich gerne.

Hätten Sie Interesse an einem unverbindlichen Testshooting?

Liebe Grüße
Sabine

intent `spam` · type `unspecified` · timeline `unspecified` · financing `None` · foreign `None` · on behalf `False` · qualified `False` · urgency `0` · action `review` · budget `None`

**Y1-31** (en, whatsapp, standard)

> hi, developer here, looking for plots in estepona / casares area for 4 to 8 townhouses, we have maybe 1.5M for the land, any off market stuff? we build and sell

intent `invest` · type `plot` · timeline `unspecified` · financing `None` · foreign `None` · on behalf `False` · qualified `True` · urgency `0` · action `send_listings` · budget `1500000`

**Y3-28** (pl, email, hard_valuation)

> Dzień dobry,

sprzedajemy nasze mieszkanie w Nueva Andalucía (3 sypialnie, Aloha), żeby kupić coś większego dla rodziny. Zanim wystawimy, prosimy o wycenę, żebyśmy wiedzieli, na jaki budżet możemy liczyć przy zakupie. Kiedy mogliby Państwo przyjechać obejrzeć? Jesteśmy dostępni po południu w tym i przyszłym tygodniu.

Pozdrawiam,
Monika

intent `sell` · type `apartment` · timeline `now` · financing `None` · foreign `None` · on behalf `False` · qualified `True` · urgency `2` · action `book_valuation` · budget `None`

**Y7-13** (fr, form, standard)

> Bonjour, je suis intéressé par la réf. 4521, la maison mitoyenne à La Cala de Mijas. Est-elle toujours disponible ? Nous serons dans la région ce week-end et aimerions la visiter samedi ou dimanche. Merci.

intent `buy` · type `townhouse` · timeline `now` · financing `None` · foreign `None` · on behalf `False` · qualified `True` · urgency `2` · action `send_listings` · budget `None`

**Y1-44** (en, voice, hard_timing)

> hi um we're flying out the second week of november, um to look at apartments in estepona, um two bed, um we can go to three hundred, three twenty tops, um we'd want to see maybe five or six places over the weekend, um we're from bristol, um we've got a mortgage in principle with halifax, um so yeah, send us what you've got

intent `buy` · type `apartment` · timeline `now` · financing `True` · foreign `True` · on behalf `False` · qualified `True` · urgency `2` · action `send_listings` · budget `320000`

**Y4-53** (es, email, standard)

> Buenas tardes,

Somos un matrimonio jubilado de Bilbao y queremos comprar un piso en planta baja o con ascensor en Fuengirola o Benalmádena, cerca del paseo marítimo, 2 dormitorios, máximo 260.000 €. Pagaríamos al contado con la venta de nuestro piso de Bilbao, que ya está firmada.

Podríamos bajar a ver pisos a partir del 3 de noviembre. Nuestro teléfono es el +44 7700 900 377.

Un saludo

intent `buy` · type `apartment` · timeline `now` · financing `False` · foreign `False` · on behalf `False` · qualified `True` · urgency `2` · action `call_now` · budget `260000`

**Y7-46** (fr, email, hard_timing)

> Bonjour,

Nous cherchons une villa à louer à l'année à Sotogrande ou Guadalmina, 4 chambres, piscine, à partir de mi-novembre. Budget 5.000 € par mois.

Vous pouvez me répondre sur m.lambert@example.com.

Cordialement,
Mathieu Lambert

intent `rent_long` · type `villa` · timeline `now` · financing `None` · foreign `None` · on behalf `False` · qualified `True` · urgency `2` · action `send_listings` · budget `5000`

**Y5-24** (de, whatsapp, standard)

> mal ne frage: kann man als deutscher ohne NIE überhaupt ein haus kaufen? wir leben in frankfurt und haben am wochenende ein haus in cancelada gesehen, 620.000, und würden gern diese woche noch ein angebot machen bevor es weg ist. ruft mich bitte an +44 7700 900 582

intent `buy` · type `villa` · timeline `now` · financing `None` · foreign `True` · on behalf `False` · qualified `True` · urgency `3` · action `call_now` · budget `620000`

**Y7-11** (fr, email, standard)

> Bonjour,

J'ai analysé votre site et vos annonces n'apparaissent pas en première page Google pour « appartement Marbella ». Notre agence SEO spécialisée immobilier garantit le top 3 en 90 jours, à partir de 490 €/mois.

Audit gratuit sur simple réponse : seo@example.com

Cordialement,
Kevin, Growth Manager

intent `spam` · type `unspecified` · timeline `unspecified` · financing `None` · foreign `None` · on behalf `False` · qualified `False` · urgency `0` · action `review` · budget `None`

