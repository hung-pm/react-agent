Mục tiêu triển khai

Xây một hệ thống để:
	•	import một codebase
	•	parse code theo AST
	•	chunk theo đơn vị logic
	•	embedding và lưu vào LanceDB
	•	lưu metadata + graph quan hệ
	•	hỗ trợ hybrid retrieval cho agent
	•	hỗ trợ incremental update khi file thay đổi
	•	trace ngược về source code
	•	có benchmark/evaluation để đo chất lượng

⸻

Nguyên tắc triển khai
	1.	Làm MVP nhưng phải version hóa từ đầu
Nếu không có parser_version, chunk_version, embedding_version thì sau này reindex rất đau.
	2.	Tách retrieval khỏi generation
Trước hết phải làm retrieval đúng, rồi mới gắn vào Gemma.
	3.	Không cố làm graph quá sâu ở phase đầu
Call graph depth 1 đã đủ giá trị cho MVP.
	4.	Luôn có degraded mode
Parse fail, embedding fail, vector fail thì vẫn phải còn keyword search.

⸻

Tổng quan phases
	•	Phase 0: Chốt spec và data contract
	•	Phase 1: Project import + indexing pipeline
	•	Phase 2: AST chunking + metadata extraction
	•	Phase 3: Embedding + LanceDB storage
	•	Phase 4: Hybrid retrieval tool
	•	Phase 5: Graph extraction + graph expansion
	•	Phase 6: Incremental update / file watcher / reindex
	•	Phase 7: LLM integration + answer generation
	•	Phase 8: Evaluation + benchmark + observability
	•	Phase 9: Hardening để dùng thật

⸻

Phase 0 — Chốt spec trước khi code

Mục tiêu

Khóa các quyết định để dev không hiểu mỗi người một kiểu.

Deliverables

0.1. Supported scope

Chốt rõ:
	•	ngôn ngữ hỗ trợ giai đoạn đầu: ví dụ Python + TypeScript
	•	loại file được index:
	•	.py, .ts, .tsx, .js
	•	loại file bỏ qua:
	•	node_modules, dist, build, .git, .venv, binary, generated code

0.2. Chunking spec

Chốt:
	•	unit chính: function, method, class
	•	class sẽ index:
	•	class signature
	•	từng method riêng
	•	function/method quá dài:
	•	split tiếp theo AST child block
	•	fallback cuối cùng mới dùng recursive text split
	•	metadata bắt buộc:
	•	repo_id
	•	branch
	•	commit_sha
	•	file_path
	•	language
	•	symbol_name
	•	symbol_type
	•	signature
	•	start_line
	•	end_line
	•	parent_symbol
	•	chunk_hash
	•	file_hash
	•	parser_version
	•	chunk_version
	•	embedding_version

0.3. Retrieval modes

Chốt 4 mode:
	•	symbol_lookup
	•	semantic_search
	•	hybrid_search
	•	graph_expand

0.4. Update events

Chốt event types:
	•	full_import
	•	file_created
	•	file_modified
	•	file_deleted
	•	file_renamed
	•	reindex_all

0.5. Context budget policy

Chốt:
	•	initial retrieve: top 20
	•	rerank còn top 8
	•	final push cho LLM: top 5
	•	không dùng quá 60% context window

⸻

Phase 1 — Repository import & indexing pipeline

Mục tiêu

Cho phép đưa project vào hệ thống và xây index đầu tiên.

Công việc

1.1. Repository scanner

Viết module:
	•	scan tree file
	•	detect language theo extension
	•	apply include/exclude rules
	•	tạo danh sách file indexable

1.2. Job orchestration

Tạo indexing job lifecycle:
	•	pending
	•	scanning
	•	parsing
	•	chunking
	•	embedding
	•	graph_building
	•	completed
	•	partial_failed
	•	failed

1.3. Repository manifest

Mỗi lần import lưu:
	•	repo_id
	•	repo_name
	•	root_path hoặc remote URL
	•	current_branch
	•	current_commit
	•	indexed_at
	•	parser/chunk/embed versions

Deliverables
	•	import được 1 repo local
	•	tạo job status
	•	scan được file hợp lệ
	•	lưu manifest repo

⸻

Phase 2 — AST parser & chunking engine

Mục tiêu

Chuyển source code thành các chunk logic ổn định.

Công việc

2.1. Parser adapter per language

Dùng Tree-sitter và tạo adapter riêng:
	•	Python parser adapter
	•	TypeScript parser adapter

Mỗi adapter phải trả về:
	•	file-level AST
	•	symbol list
	•	call candidates
	•	parse diagnostics

2.2. Symbol extraction

Tách:
	•	function
	•	method
	•	class
	•	constructor
	•	module-level constants quan trọng nếu cần

2.3. Chunk creation rules

Rule cụ thể:
	•	function/method bình thường = 1 chunk
	•	method trong class lưu cả:
	•	class_name
	•	method_name
	•	full signature
	•	class có thể thêm 1 summary chunk nếu class lớn
	•	function quá dài:
	•	split theo AST child block
	•	nếu vẫn dài thì fallback recursive split theo \n\n, \n, ;

2.4. Fallback strategy

Nếu parse fail:
	•	tạo raw text chunks theo line window
	•	đánh dấu parse_status=failed

Deliverables
	•	parse được file
	•	sinh chunks ổn định
	•	có line range đúng
	•	có metadata đầy đủ

⸻

Phase 3 — Storage layer: LanceDB + metadata store + graph store

Mục tiêu

Lưu embeddings, metadata và graph sao cho update được.

Quyết định kiến trúc

Tôi không nghĩ nên nhét mọi thứ vào mỗi LanceDB.

Nên tách:
	•	LanceDB: embeddings + metadata retrieval fields
	•	SQLite/Postgres: repo/file/chunk/index jobs
	•	SQLite/Postgres graph_edges: edges caller/callee
	•	MVP có thể dùng SQLite
	•	production có thể lên Postgres

Công việc

3.1. Tables / collections

Thiết kế các bảng chính:

repositories
	•	repo_id
	•	repo_name
	•	branch
	•	latest_commit_sha
	•	status
	•	created_at
	•	updated_at

files
	•	file_id
	•	repo_id
	•	file_path
	•	language
	•	file_hash
	•	current_commit_sha
	•	parse_status
	•	updated_at

chunks
	•	chunk_id
	•	repo_id
	•	file_id
	•	file_path
	•	symbol_name
	•	symbol_type
	•	parent_symbol
	•	signature
	•	start_line
	•	end_line
	•	content
	•	content_summary optional
	•	chunk_hash
	•	file_hash
	•	commit_sha
	•	parser_version
	•	chunk_version
	•	embedding_version
	•	is_active

graph_edges
	•	edge_id
	•	repo_id
	•	source_symbol
	•	target_symbol
	•	source_file
	•	target_file
	•	edge_type
	•	confidence
	•	commit_sha

indexing_jobs
	•	job_id
	•	repo_id
	•	trigger_type
	•	status
	•	stats_json
	•	error_json
	•	started_at
	•	ended_at

3.2. LanceDB schema

Mỗi record vector gồm:
	•	chunk_id
	•	repo_id
	•	file_path
	•	language
	•	symbol_name
	•	symbol_type
	•	signature
	•	start_line
	•	end_line
	•	content
	•	commit_sha
	•	embedding vector

3.3. Upsert/delete policy
	•	không update từng vector lẻ kiểu ad-hoc
	•	delete theo:
	•	repo_id + file_path + commit/version cũ
	•	insert lại full chunks của file đó

Deliverables
	•	lưu được metadata
	•	lưu được vector
	•	query được theo repo/language/file_path/symbol_name
	•	xóa và insert lại được theo file

⸻

Phase 4 — Embedding pipeline

Mục tiêu

Sinh embedding nhất quán cho chunk và query.

Công việc

4.1. Embedding service wrapper

Tạo service chuẩn:
	•	embed_documents(chunks[])
	•	embed_query(query)

Dùng:
	•	nomic-embed-code

4.2. Input normalization

Trước khi embed chunk:
	•	ghép content theo format nhất quán, ví dụ:
	•	file path
	•	symbol signature
	•	code content
	•	optional short docstring/comment

Không nên chỉ embed raw code trần.

4.3. Versioning

Lưu:
	•	embedding_model_name
	•	embedding_model_version
	•	prompt format version

4.4. Failure handling

Nếu embed fail:
	•	retry N lần
	•	nếu vẫn fail:
	•	giữ chunk metadata
	•	đánh dấu embedding_status=failed
	•	fallback keyword-only

Deliverables
	•	embed được batch chunks
	•	embed query được
	•	retry/fallback hoạt động

⸻

Phase 5 — Hybrid retrieval tool

Mục tiêu

Cho agent tìm code theo nhiều kiểu query thật sự dùng được.

Công việc

5.1. Retrieval API contract

Thiết kế API/tool input:
	•	query: string
	•	repo_id: string
	•	branch: optional
	•	top_k: int
	•	language: optional
	•	symbol_types: optional
	•	file_path_filter: optional
	•	mode: auto | symbol_lookup | semantic | hybrid
	•	include_graph_context: bool

Output:
	•	results[]
	•	chunk_id
	•	file_path
	•	symbol_name
	•	symbol_type
	•	signature
	•	start_line
	•	end_line
	•	score
	•	match_type
	•	content
	•	trace
	•	commit_sha
	•	repo_id

5.2. Query analysis

Phân loại query:
	•	giống tên symbol/file => ưu tiên exact
	•	mang intent tự nhiên => semantic
	•	mơ hồ => hybrid

5.3. Exact search layer

Không chỉ metadata filter, mà phải có:
	•	symbol name exact
	•	prefix/fuzzy symbol name
	•	filepath fuzzy
	•	keyword BM25/full-text
	•	metadata filter:
	•	repo
	•	language
	•	extension

5.4. Vector search layer
	•	embed query
	•	vector search top N trong LanceDB
	•	prefilter theo repo/language nếu có

5.5. Fusion + reranking

Gộp exact + vector candidates, rồi rerank theo:
	•	symbol name overlap
	•	filepath overlap
	•	semantic score
	•	signature similarity
	•	keyword hits
	•	recency/version consistency

Deliverables
	•	query “hàm xử lý login” ra được code liên quan
	•	query exact symbol như processPayment ra đúng symbol
	•	hybrid hoạt động tốt hơn vector-only

⸻

Phase 6 — Graph extraction & graph-based context

Mục tiêu

Giải quyết case semantic search tìm đúng hàm nhưng thiếu ngữ cảnh caller/callee.

Công việc

6.1. Graph extraction

Từ AST, build edges:
	•	calls
	•	defined_in
	•	belongs_to_class
	•	inherits_from nếu làm được ở phase đầu
	•	imports optional

MVP nên bắt đầu bằng:
	•	calls
	•	belongs_to_class

6.2. Edge confidence

Vì static analysis không luôn đúng, nhất là Python/JS:
	•	edge nào chắc thì confidence cao
	•	unresolved/dynamic call confidence thấp

6.3. Graph traversal service

API:
	•	given symbol_name hoặc chunk_id
	•	lấy:
	•	callers depth 1
	•	callees depth 1
	•	optional depth 2
	•	giới hạn max nodes

6.4. Graph expansion policy

Chỉ expand khi:
	•	query hỏi flow/call/usage/dependency
	•	semantic result chưa đủ context
	•	agent đang debug upstream/downstream logic

Default:
	•	depth = 1
	•	max_nodes = 6 hoặc 8

Deliverables
	•	biết hàm nào gọi hàm nào
	•	mở rộng caller/callee được
	•	trả context liên quan mà không nổ quá nhiều noise

⸻

Phase 7 — Incremental update & change triggers

Mục tiêu

Khi file thay đổi, index cập nhật đúng mà không phải reindex cả repo mỗi lần.

Công việc

7.1. Change detection

Hỗ trợ trigger từ:
	•	manual reindex
	•	git diff
	•	file watcher
	•	IDE/plugin save event

7.2. Event handling

Cho từng loại event:

file_created
	•	parse file mới
	•	chunk
	•	embed
	•	insert metadata/vector/edges

file_modified
	•	tìm file cũ theo repo_id + file_path
	•	invalidate chunks cũ
	•	xóa vector cũ
	•	rebuild chunks mới
	•	rebuild edges liên quan file

file_deleted
	•	mark file inactive
	•	delete/inactivate chunks
	•	delete edges liên quan

file_renamed
	•	xử lý như delete old + create new
	•	hoặc migrate metadata nếu giữ history

7.3. Debounce & batching

Vì save liên tục sẽ spam pipeline:
	•	debounce 1–3 giây
	•	batch nhiều file cùng lúc

7.4. Reindex all

Bắt buộc có khi:
	•	parser_version đổi
	•	chunking strategy đổi
	•	embedding_version đổi

Deliverables
	•	sửa 1 file thì chỉ reindex file đó
	•	delete/rename không để lại vector rác
	•	có full reindex khi schema/version đổi

⸻

Phase 8 — Context builder cho LLM

Mục tiêu

Đóng gói retrieval result thành prompt context sạch, có trace, không quá budget.

Công việc

8.1. Context packing

Sau retrieval:
	•	deduplicate chunk
	•	merge chunk liền kề cùng file nếu cần
	•	ưu tiên chunk có score cao và ít trùng lặp
	•	cắt theo token budget

8.2. Trace injection

Mỗi context chunk nên kèm:
	•	file path
	•	line range
	•	symbol name
	•	commit sha

8.3. Prompt template

Prompt cho Gemma nên gồm:
	•	user query
	•	retrieved code contexts
	•	instructions:
	•	chỉ trả lời dựa trên context
	•	nếu không chắc thì nói thiếu context
	•	luôn trích nguồn file/line

8.4. Answer schema

Response nên có:
	•	answer
	•	cited_sources[]
	•	maybe_followup_queries optional

Deliverables
	•	agent trả lời có dẫn source
	•	không vượt context budget
	•	hạn chế hallucination

⸻

Phase 9 — LLM integration

Mục tiêu

Gắn Gemma-4-e2b-it qua LM Studio API vào pipeline thật.

Công việc

9.1. LLM wrapper

Tạo client:
	•	chat completion
	•	low temperature 0.1–0.2
	•	timeout/retry
	•	max tokens

9.2. Answer modes

Hỗ trợ ít nhất:
	•	code explanation
	•	symbol lookup answer
	•	debug trace answer
	•	dependency explanation

9.3. Guardrails

Nếu retrieval score thấp:
	•	đừng trả lời quá tự tin
	•	yêu cầu thêm context hoặc trả “không đủ chắc chắn”

Deliverables
	•	truy vấn end-to-end được
	•	câu trả lời có source mapping

⸻

Phase 10 — Evaluation & benchmark

Mục tiêu

Biết hệ thống đang cải thiện thật hay chỉ “cảm giác tốt hơn”.

Công việc

10.1. Gold dataset

Tạo benchmark từ:
	•	commit history
	•	bug fixes
	•	PR discussions
	•	tài liệu team biết sẵn

Mỗi sample nên có:
	•	query
	•	expected file(s)
	•	expected symbol(s)
	•	task_type
	•	difficulty
	•	notes

10.2. Retrieval metrics

Đo:
	•	Recall@K
	•	Precision@K
	•	MRR
	•	Hit@1
	•	symbol hit rate
	•	file hit rate

10.3. End-to-end RAG metrics

Dùng:
	•	RAGAS
	•	TruLens

Đo:
	•	faithfulness
	•	answer relevance
	•	context precision

10.4. Regression suite

Mỗi lần đổi:
	•	chunking
	•	reranking
	•	embedding model
	•	graph depth

thì phải chạy benchmark lại.

Deliverables
	•	dashboard hoặc report benchmark
	•	so sánh version trước/sau

⸻

Phase 11 — Observability & debugging

Mục tiêu

Khi hệ thống sai, biết sai ở đâu.

Công việc

11.1. Logging

Log theo từng bước:
	•	query normalize
	•	retrieval candidates
	•	vector scores
	•	exact matches
	•	rerank results
	•	graph expansion results
	•	final packed context

11.2. Explainability

Cho dev xem được:
	•	vì sao chunk này được chọn
	•	nó đến từ exact hay vector
	•	graph edge nào kéo nó vào

11.3. Error reporting

Phân biệt rõ:
	•	parser fail
	•	embed fail
	•	vector search fail
	•	graph fail
	•	LLM fail

Deliverables
	•	debug một query cụ thể được
	•	trace full pipeline được

⸻

Phase 12 — Hardening để dùng production

Mục tiêu

Biến hệ thống từ prototype sang usable thật.

Công việc

12.1. Performance
	•	batch embedding
	•	caching query embeddings
	•	ANN tuning trong LanceDB
	•	parallel parse/chunk/embed

12.2. Data hygiene
	•	cleanup old inactive records
	•	retention policy theo commit/version
	•	reindex scheduler

12.3. Security
	•	isolate repo theo tenant/user
	•	không index secret files
	•	filter .env, keys, credentials nếu cần

12.4. Multi-branch
	•	branch-aware retrieval
	•	mặc định branch hiện tại
	•	không lẫn source giữa các branch

Deliverables
	•	hệ thống ổn định hơn
	•	ít rác dữ liệu
	•	retrieval đúng branch/repo

⸻

Phân rã theo workstream cho team

Workstream A — Parsing & Chunking

Làm:
	•	Tree-sitter adapters
	•	symbol extraction
	•	chunking rules
	•	fallback lexical chunking

Workstream B — Storage & Indexing

Làm:
	•	metadata DB schema
	•	LanceDB schema
	•	indexing job engine
	•	full import / incremental update

Workstream C — Retrieval

Làm:
	•	exact search
	•	vector search
	•	hybrid fusion
	•	rerank
	•	context budgeter

Workstream D — Graph

Làm:
	•	graph extraction
	•	edge storage
	•	traversal API
	•	graph expansion policy

Workstream E — LLM & Evaluation

Làm:
	•	Gemma integration
	•	answer prompt
	•	traceability
	•	benchmark
	•	regression tests

⸻

Milestone thực tế

Milestone 1 — Search được code theo semantic + exact

Bao gồm:
	•	import repo
	•	chunk AST
	•	embed LanceDB
	•	hybrid retrieval
	•	trả source path + line

Milestone 2 — Update được khi file đổi

Bao gồm:
	•	file modified/create/delete
	•	invalidate + reindex đúng file
	•	không cần reindex full repo

Milestone 3 — Graph context cho debug

Bao gồm:
	•	call graph depth 1
	•	caller/callee expansion
	•	context packing tốt hơn cho debug

Milestone 4 — End-to-end agent answer

Bao gồm:
	•	retrieval -> pack context -> Gemma answer
	•	traceable answer
	•	benchmark baseline

⸻

Acceptance criteria tổng

Hệ thống được coi là “đủ đầy đủ” khi đạt tối thiểu:
	•	import được repo và index thành công
	•	query tự nhiên tìm được top code chunks liên quan
	•	query exact symbol trả đúng file/symbol
	•	file thay đổi thì reindex chỉ phần bị ảnh hưởng
	•	answer có trace về file + dòng
	•	parser/embedding fail vẫn còn degraded mode
	•	benchmark retrieval có số đo rõ ràng
	•	mỗi lần đổi chunking/model đều so sánh lại được

⸻

Thứ tự triển khai tôi khuyên dùng

Đừng làm graph quá sớm. Thứ tự an toàn là:
	1.	spec + schema
	2.	import + parser + chunking
	3.	embedding + LanceDB
	4.	exact + vector + hybrid retrieval
	5.	context packing
	6.	Gemma integration
	7.	incremental update
	8.	graph expansion
	9.	evaluation + hardening