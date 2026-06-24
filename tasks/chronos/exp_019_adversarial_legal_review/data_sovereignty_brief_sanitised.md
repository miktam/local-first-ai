---
version: v_final
derived_from: adversarial panel review (Regulator, Opposing Counsel, Devil's Advocate)
date: 2026-06-23
status: DRAFT — for discussion with qualified legal counsel only
sanitised: true
---

# Data Sovereignty Legal Brief
## Cloud AI versus Local Inference for Document Processing

**DRAFT — For discussion with qualified legal counsel only. This document does not constitute legal advice and should not be relied upon as such.**

*Prepared by: Local inference operator*
*Primary jurisdiction: Gibraltar (UK-aligned)*
*Secondary: Singapore · Tertiary: EU*
*Date: 2026-06-23*

---

## Part 1 — Preliminary: The Three Data States and the Architectural Requirement

Data exists in three technical states: at rest (stored, not being processed), in transit (moving across a network), and in use (actively being computed upon). Existing legal frameworks address the first two states with established mechanisms. The third state — data in use — is where the specific problems this brief addresses arise.

When a document is submitted to a remote AI provider's API for inference, it is in use on infrastructure outside the operator's control for the duration of that processing call. It is this moment that creates the CLOUD Act exposure and cross-border transfer obligations analysed below.

**Architectural requirement — co-location:** The central claim of this brief — that local inference avoids international data transfers during processing — holds only if the inference hardware is in the same jurisdiction as, or the same physical location as, the documents being processed. If documents originate in Gibraltar and are processed by a cluster located in Singapore, the documents must travel from Gibraltar to Singapore before inference can occur. That transit is itself an international transfer, triggering the Gibraltar/UK GDPR restrictions and Singapore PDPA obligations described below. This is not a minor caveat: it is a precondition for the analysis in this brief to apply. **Any deployment in which the compute cluster is in a different jurisdiction from the document corpus must separately address the cross-border transfer of documents to the cluster.** The analysis below assumes compute and data are co-located.

**Control plane caveat:** Co-location of compute and data is necessary but not sufficient. If the local hardware relies on an overseas or US-headquartered control plane — for remote orchestration, centralised logging, hybrid cloud management, or vendor-initiated diagnostic access — a restricted transfer or extraterritorial jurisdiction vector may still be triggered even where the inference hardware is physically present in the target jurisdiction. The analysis below assumes no such dependency exists.

---

## Part 2 — The CLOUD Act

**Issue:** Does processing documents through a US-headquartered AI provider's API create US legal exposure, regardless of server location?

**Law:** The Clarifying Lawful Overseas Use of Data Act (CLOUD Act), 18 U.S.C. § 2713, enacted March 2018, requires providers "subject to United States jurisdiction" to preserve and disclose data "in the provider's possession, custody, or control" when served with a valid US legal process, "regardless of whether such communication, record, or other information is located within or outside of the United States." A provider is subject to US jurisdiction if incorporated, headquartered, or having its principal place of business in the United States. All major commercial AI API providers — OpenAI, Anthropic, Google (Gemini API), Microsoft (Azure AI), and Amazon (Bedrock) — satisfy this criterion.

**CLAIM-001 — CLOUD Act applies regardless of server location:**

The CLOUD Act's extraterritorial reach is unambiguous: server geography is irrelevant. The statute follows the corporate structure of the provider. A US court order served on a provider's US-incorporated entity can compel disclosure of data held or processed anywhere in the world.

*Conclusion:* This is established law. **SOLID.**

**CLAIM-002 — Transient inference as "possession":**

It is arguable that when a document is submitted to a US AI provider's API for inference, the document is, for the duration of that processing window, in the "possession, custody, or control" of a US-jurisdiction provider within the meaning of 18 U.S.C. § 2713. This argument is legally credible. However, whether transient RAM-resident processing — where a document is processed and not retained — satisfies the "possession" threshold has not been definitively tested by a US court. This is the most material legal uncertainty in this brief and should be treated as an open question pending authoritative guidance.

An additional factual matter affects this analysis: many AI providers retain logs, temporary cache, or inference records for abuse detection, rate-limiting, or model improvement purposes. If a provider retains such records, the "transient" characterisation does not apply — the data is stored, and CLOUD Act possession is unambiguous. **The specific logging and data retention practices of any AI provider relied upon must be verified before any reliance is placed on a transient-possession argument.**

*Conclusion:* The CLOUD Act likely applies to AI inference through a US provider's API; transient inference as a specific category of "possession" is arguable but untested. **CONTESTED — treat as contested pending legal guidance and provider-specific factual verification.**

**CLAIM-003 — Local inference removes the US CLOUD Act vector:**

Where inference is performed on hardware operated by a non-US entity (the operator, with no US parent company or US-incorporated affiliate), using an open-source model whose weights reside on-device, and no document content is transmitted to any external endpoint during inference, no US-jurisdiction entity has possession of document content during the processing stage. The CLOUD Act has no US-entity addressable party for the inference stage.

This exclusion is structural for the inference stage specifically. It does not prevent legal process served on the operator itself — the operator, as the entity in possession of both the documents and the inference infrastructure, remains an addressable party under the laws of its own jurisdiction(s). A court with jurisdiction over the operator can compel the operator to produce documents regardless of the architecture.

*Note on the software supply chain:* The Ollama runtime (US-origin open-source software) serves the model locally. Ollama's binary does not transmit document content to external servers during inference. This should be verified by reference to Ollama's open-source code and privacy policy. The Gemma 4 model weights are distributed by Google DeepMind as open-source; once downloaded to local storage, Google has no access to documents processed using those weights.

*Conclusion:* Local inference removes the US CLOUD Act vector specifically for the inference stage, provided the operator has no US parent and no document content reaches any external endpoint during processing. **SOLID — with the vendor-structure and Ollama telemetry caveats above.**

### The UK-US Bilateral Data Access Agreement (BDAA)

A material qualification to the CLOUD Act analysis — relevant specifically for Gibraltar and UK-aligned entities — is the UK-US Bilateral Data Access Agreement (BDAA), which came into force on 3 October 2022. The BDAA is an executive agreement made pursuant to 18 U.S.C. § 2523 (the CLOUD Act provision authorising such bilateral instruments). Its full title is *Agreement between the Government of the United Kingdom of Great Britain and Northern Ireland and the Government of the United States of America on Access to Electronic Data for the Purpose of Countering Serious Crime*.

The BDAA:
- Enables UK law enforcement to directly request data from US companies without the slower MLAT process
- **Establishes a legal safeguards framework governing US law enforcement requests for data on UK persons from US companies, including procedural requirements and protections for UK data subjects**
- Creates agreed channels under which US authorities must route requests for UK persons' data

For Gibraltar/UK-aligned entities and UK data subjects specifically, the BDAA reduces the practical CLOUD Act risk from US AI providers for **law enforcement access scenarios**. It does not, however, address:
- Civil litigation compelled disclosure (subpoenas in private proceedings)
- National security orders (which operate under separate statutory authority)
- Intelligence-community access

**The practical effect for this brief:** The CLOUD Act risk from a US AI provider, for law enforcement access to Gibraltar/UK persons' data specifically, is materially constrained by the BDAA. The residual concern — civil subpoena and intelligence access — remains. Legal counsel should advise on the relative likelihood of each vector for the operator's specific use case.

---

## Part 3 — Jurisdiction-by-Jurisdiction Analysis

### A. Gibraltar (Primary Jurisdiction)

**Issue:** Under Gibraltar's Data Protection Act 2004 and its post-Brexit UK GDPR alignment, does submitting documents to a US AI provider's API constitute a restricted international transfer?

**Law:** Gibraltar's Data Protection Act 2004 was updated post-Brexit to reflect UK GDPR principles. The Gibraltar Regulatory Authority (GRA) is the supervisory authority for data protection. The GRA operates independently and is not bound by ICO decisions, though it is understood to apply standards broadly consistent with the UK GDPR framework. **Gibraltar-qualified counsel should confirm the GRA's specific position on the questions below; the analysis that follows is submitted on the basis of GRA alignment with UK GDPR principles, subject to that confirmation.**

Under UK GDPR Article 44 (as reflected in Gibraltar's framework), transfers of personal data to a third country require either an adequacy decision, appropriate safeguards (UK IDTA or UK Addendum to EU SCCs), or a specific derogation. It is submitted, on the basis of the UK ICO's post-Schrems II guidance on restricted transfers, that transient processing of personal data on third-country infrastructure constitutes a restricted transfer — the analysis turning on whether a legally distinct overseas recipient can access the data, rather than whether data is stored abroad after the call. This position should be confirmed by reference to a specific ICO publication and, given the GRA's distinct regulatory position, by Gibraltar-qualified counsel before reliance is placed on it. See Open Question 3.

**Application — cloud inference:** A Gibraltar entity submitting documents to a US AI API is engaged in a restricted international transfer (subject to GRA confirmation of the transient-processing position). The absence of a UK-US adequacy decision means the transfer requires a UK IDTA supplemented by a Transfer Risk Assessment. That TRA must address CLOUD Act compelled-disclosure exposure — noting however the BDAA qualification above for law enforcement scenarios.

**Application — local inference:** Where compute is co-located with the documents (see Part 1 co-location requirement), no international transfer occurs during inference. Transfer restrictions are not triggered.

*Conclusion on CLAIM-006:* Transient processing by a US AI API is submitted to constitute a restricted transfer under Gibraltar law, subject to GRA confirmation. Local inference with co-located compute avoids this trigger. **CONTESTED pending GRA-specific confirmation; directionally correct.**

---

### B. Singapore (Secondary Jurisdiction)

**Issue:** Under the Singapore PDPA and MAS TRM Guidelines, what are the compliance implications of using a US cloud AI API?

**Law — PDPA:** Section 26 of the PDPA (Cap. 26G, as amended 2020) requires that personal data not be transferred outside Singapore unless the recipient provides comparable protection. It is submitted that real-time processing of personal data by a foreign entity — including AI inference — constitutes a transfer for the purposes of Section 26, on the basis that the data is disclosed to and processed by an entity outside Singapore. This interpretation should be confirmed by reference to the PDPC's current advisory guidelines and, if necessary, formal PDPC guidance on AI inference specifically.

**Law — MAS Outsourcing Guidelines:** The MAS Guidelines on Outsourcing (most recently revised October 2018) define material outsourcing by reference to the significance of the outsourced service to the institution's business operations, regulatory compliance obligations, and risk profile. The Guidelines also require prior notification to MAS before entering into material outsourcing arrangements. The specific paragraph references should be confirmed with MAS-experienced counsel. Whether use of a US AI API for document processing constitutes material outsourcing is a facts-dependent assessment; for a MAS-regulated PE firm using such a service to process investment data, portfolio company documents, or investor information, the threshold is likely met. See Open Question 6.

**Critical distinction — the operator as operator vs. the operator as vendor:**

- If a MAS-regulated entity operates local inference infrastructure itself (in-house), there is no outsourcing and MAS TRM outsourcing obligations do not apply to the inference stage.
- If the operator provides document processing as a service to a MAS-regulated entity, the operator is the outsourced provider. MAS TRM outsourcing obligations apply to the relationship between the regulated entity and the operator — regardless of whether the operator uses local inference internally. The benefit of local inference, in this scenario, is that the operator presents a more auditable and regulatorily defensible outsourcing arrangement than a US hyperscaler — not that outsourcing obligations are eliminated.

**Application:** For a MAS-regulated entity using a US AI API directly: PDPA Section 26 transfer obligations are triggered; material outsourcing requirements likely apply including MAS notification. For a MAS-regulated entity using the operator's local-inference service: outsourcing relationship is with the operator (not a US entity); the operator is auditable; PDPA transfer obligations depend on where the operator's cluster is located relative to Singapore.

*Conclusion on CLAIM-004 and CLAIM-005:* PDPA transfer obligations are submitted to apply to AI inference through a US API. MAS TRM material outsourcing obligations likely apply. Local inference eliminates the US-provider outsourcing risk but does not eliminate outsourcing obligations where the operator is a third-party vendor. **CONTESTED on CLAIM-004 (PDPC citation to be verified); SOLID-with-caveats on CLAIM-005 once vendor/in-house distinction is applied.**

---

### C. European Union (Tertiary — EU Counterparties)

**Issue:** For EU personal data or EU-based counterparties, does GDPR impose additional obligations?

**Law:** GDPR Articles 44–49 restrict transfers of personal data outside the EEA. Post-Schrems II (C-311/18, July 2020), transfers to the US require SCCs plus a Transfer Impact Assessment addressing US surveillance law exposure. The EU-US Data Privacy Framework (adequacy decision, July 2023) provides an alternative for DPF-certified providers, but faces ongoing legal challenges before the CJEU and should be treated as provisional.

**Application — cloud AI:** EU personal data submitted to a US AI API requires SCCs and a TIA addressing CLOUD Act exposure. The CLOUD Act is the central risk factor in TIAs for US AI providers. The BDAA does not apply to EU-US transfers (it is a UK-US instrument only).

**Application — local inference, critical caveat:** Local inference on a Gibraltar-based cluster avoids the EU-to-US transfer for the US extraterritorial access risk specifically. However, **Gibraltar is not covered by the EU-UK Brexit adequacy decision.** The EU adequacy decision applies to the United Kingdom proper; Gibraltar is a British Overseas Territory and is not included within its scope for GDPR Chapter V purposes. Accordingly, any transfer of EU personal data from the EEA to a Gibraltar-based inference cluster is a restricted international transfer requiring its own compliance mechanism — SCCs or equivalent — independent of and in addition to the US AI question this brief primarily addresses.

*Conclusion:* EU personal data processed through a US AI API requires compliance mechanisms that are difficult to satisfy in several EU jurisdictions. Local inference in Gibraltar eliminates the US cloud provider access risk for EU data, but does not eliminate the EEA-to-Gibraltar transfer requirement. **CONTESTED — revised from SOLID; Gibraltar adequacy gap is material for EU counterparties. Requires EU/Gibraltar-qualified counsel.**

---

## Part 4 — What the Structural Mitigation Means (and Does Not Mean)

*Note on terminology: "structural guarantee" has been revised to "structural mitigation" throughout. No technical architecture provides an absolute legal shield. A tech stack can be compromised by supply-chain vulnerabilities, firmware updates, or undisclosed vendor telemetry. The correct framing is risk mitigation, not immunity.*

**What the structural mitigation achieves — specifically:**

During inference, document content does not leave the hardware perimeter (subject to the control plane caveat in Part 1). No US-jurisdiction entity has possession of document content during processing. CLOUD Act compelled disclosure has no US-entity addressable party for the inference stage (subject to the US nexus scenarios noted in Part 2). No international transfer of document content occurs during inference (where compute and data are co-located). Cross-border transfer restriction mechanisms are not required for the processing stage itself — subject to this analysis and its caveats.

**What it does not achieve:**

1. **The operator remains an addressable party.** The operator, as the entity in possession of documents and infrastructure, can be served with court orders, search warrants, regulatory demands, and letters rogatory under the laws of its own jurisdiction. The structural mitigation prevents remote access by a US cloud provider; it does not prevent lawful compelled disclosure from the operator itself.

2. **Data protection obligations are not eliminated.** The operator remains a data controller and/or processor bearing all applicable obligations — access controls, encryption at rest, incident response, breach notification — under Gibraltar, Singapore, EU, and other applicable law.

3. **Security risk is redistributed, not eliminated.** US hyperscalers operate with SOC 2 Type II certification, ISO 27001, 24/7 security operations, and professional incident response infrastructure. Local inference on privately operated hardware accepts responsibility for security controls that would otherwise be managed by a professional cloud operator. This is a legitimate architectural choice; it should be made consciously, with appropriate compensating controls implemented.

4. **The mitigation covers inference only.** Storage, access, transmission, and all other stages of document handling remain subject to applicable law independent of the inference architecture.

**Summary framing for legal counsel:** Local inference materially reduces and bounds data sovereignty exposure for the inference stage. It removes the US CLOUD Act vector for that stage where the operator has no US nexus. It does not grant immunity from all legal process, does not resolve the EEA-to-Gibraltar transfer requirement for EU data, and shifts security responsibility to the operating entity.

---

## Part 5 — Open Questions for Legal Counsel

1. **Transient possession under the CLOUD Act.** Whether 18 U.S.C. § 2713 applies to data present on US-provider infrastructure only during active inference (not stored) is untested. This is the most material legal uncertainty in this brief.

2. **AI provider data retention.** Whether a specific US AI provider retains any logs, cache, or temporary records during inference is a factual matter that must be verified per provider. Transient-possession arguments collapse if the provider retains records.

3. **Gibraltar GRA confirmation.** Whether the GRA applies the ICO's post-Schrems II position on transient processing as a restricted transfer requires confirmation from Gibraltar-qualified counsel and, ideally, a GRA-specific source.

4. **UK-US BDAA scope.** The BDAA reduces CLOUD Act risk for law enforcement scenarios involving UK/Gibraltar persons' data. Its application to civil subpoena and intelligence-community access requires legal opinion. BDAA scope should be confirmed with UK-qualified counsel before any reliance is placed on it as a risk mitigation.

5. **PDPC AI inference guidance.** Whether the PDPC's Advisory Guidelines specifically characterise AI inference as a cross-border transfer under Section 26 PDPA should be confirmed by reference to current PDPC guidance.

6. **MAS TRM materialisation threshold.** Whether specific AI processing arrangements constitute material outsourcing requiring prior MAS notification is a facts-dependent assessment requiring MAS-experienced counsel.

7. **Ingestion transfer.** If the inference cluster is in a different jurisdiction from the document corpus, the cross-border transfer of documents to the cluster is a separate compliance matter not addressed by this brief. This must be resolved architecturally (co-location) or legally (separate transfer compliance analysis) before deployment.

8. **Ollama telemetry.** The Ollama runtime is US-origin open-source software. Confirmation that no document content or metadata is transmitted to Ollama's servers during inference should be obtained by code review of the relevant Ollama version and verified against Ollama's privacy documentation.

9. **the operator corporate structure.** This analysis assumes the operator has no US parent, no US-incorporated affiliate, and no US-based personnel with authority over the inference infrastructure. Any change requires the CLOUD Act analysis to be revisited.

10. **Existing cloud posture.** If the operator or the instructing entity currently uses US-headquartered SaaS platforms (Microsoft 365, Google Workspace, Dropbox, etc.), CLOUD Act exposure already exists for data processed on those platforms. This brief analyses the incremental exposure from AI inference; the aggregate exposure must be assessed in the context of the full technology stack.

11. **MAS Outsourcing Guidelines paragraph references.** The specific paragraph numbers for the material outsourcing definition and prior notification requirement in the MAS Guidelines on Outsourcing should be confirmed with MAS-experienced counsel.

12. **Gibraltar-EEA adequacy mechanism.** Gibraltar is not covered by the EU-UK Brexit adequacy decision. Any transfer of EU personal data from the EEA to a Gibraltar-based inference cluster is a restricted transfer requiring its own compliance mechanism (SCCs or equivalent). This question must be resolved before the analysis in Part 3C is relied upon for EEA-originating data.

---

*DRAFT — For discussion with qualified legal counsel only. This document does not constitute legal advice and should not be relied upon as such. Every open question in Part 5 should be addressed by qualified counsel before any reliance is placed on this brief.*
