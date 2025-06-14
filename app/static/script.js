document.addEventListener('DOMContentLoaded', function() {
    const chatArea = document.getElementById('chat-area');
    const userInput = document.getElementById('user-input');
    const sendBtn = document.getElementById('send-btn');
    const uploadBtn = document.getElementById('upload-btn');
    const fileUploadOptions = document.getElementById('file-upload-options');
    const closeUploadBtn = document.getElementById('close-upload');
    const typingIndicator = document.getElementById('typing-indicator');
    let conversationHistory = []; // 대화 기록 (필요시 LLM에 전달)

    // 채팅 스크롤 항상 하단 유지
    function scrollToBottom() {
        chatArea.scrollTop = chatArea.scrollHeight;
    }

    // HTML 태그 이스케이프 함수 (XSS 방지)
    function escapeHtml(text) {
        if (typeof text !== 'string') {
            console.warn("escapeHtml: 입력값이 문자열이 아닙니다. 문자열로 변환합니다.", text);
            text = String(text);
        }
        return text.replace(/[&<>"']/g, function(m) {
            return ({
                '&': '&amp;',
                '<': '&lt;',
                '>': '&gt;',
                '"': '&quot;',
                "'": '&#039;'
            })[m];
        });
    }

    // 사용자 메시지를 채팅창에 추가하는 함수
    function addUserMessage(message) {
        const messageElement = `
            <div class="flex items-start space-x-3 justify-end my-2">
                <div class="max-w-xs md:max-w-md lg:max-w-lg bg-gray-200 rounded-xl p-3 shadow">
                    <p class="font-medium text-gray-800">사용자</p>
                    <p class="text-gray-700 mt-1 whitespace-pre-wrap">${escapeHtml(message)}</p>
                </div>
                <div class="bg-gray-300 text-gray-800 rounded-full w-10 h-10 flex items-center justify-center flex-shrink-0">
                    <i class="fas fa-user"></i>
                </div>
            </div>
        `;
        chatArea.insertAdjacentHTML('beforeend', messageElement);
        scrollToBottom();
    }

    // AI 메시지를 채팅창에 추가하는 함수
    function addAiMessage(message, isAlert = false) {
        // 메시지 내의 줄바꿈(\n)을 <br> 태그로 변경하고, **텍스트**를 <strong>으로 변경
        let formattedMessage = escapeHtml(message)
                                .replace(/\n/g, '<br>')
                                .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');

        const alertBgClass = isAlert ? 'bg-red-50 border-l-4 border-red-500' : 'bg-blue-50';
        const titleColorClass = isAlert ? 'text-red-800 font-bold' : 'text-blue-800 font-medium';
        const iconBgClass = isAlert ? 'bg-red-100 text-red-800' : 'bg-blue-100 text-blue-800';


        const messageElement = `
            <div class="flex items-start space-x-3 my-2">
                <div class="${iconBgClass} rounded-full w-10 h-10 flex items-center justify-center flex-shrink-0">
                    <i class="fas fa-robot"></i>
                </div>
                <div class="max-w-xs md:max-w-md lg:max-w-lg ${alertBgClass} rounded-xl p-3 shadow">
                    <p class="${titleColorClass}">보이스피싱 방지 AI</p>
                    <p class="text-gray-700 mt-1 whitespace-pre-wrap">${formattedMessage}</p>
                </div>
            </div>
        `;
        chatArea.insertAdjacentHTML('beforeend', messageElement);
        scrollToBottom();
    }

    // 일반 텍스트 메시지 전송 처리 함수
    async function sendMessage() {
        const message = userInput.value.trim();
        if (message === '') return;

        addUserMessage(message);
        // conversationHistory.push({role: "user", content: message}); // 대화 기록 저장 (필요시)
        userInput.value = ''; // 입력창 비우기
        typingIndicator.classList.remove('hidden'); // 로딩 인디케이터 표시
        scrollToBottom();

        try {
            const response = await fetch('/chat', { // Flask의 /chat 라우트 호출
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    message: message,
                    // conversation_history: conversationHistory // 이전 대화 기록 전달 (필요시)
                })
            });

            if (!response.ok) { // 서버 응답이 정상이 아닐 경우
                const errData = await response.json().catch(() => ({ error: `서버 응답 오류 (${response.status})` }));
                throw new Error(errData.error || `서버 오류 (${response.status})`);
            }

            const data = await response.json(); // 정상 응답 JSON 파싱
            addAiMessage(data.message || "AI로부터 응답을 받지 못했습니다."); // AI 응답 메시지 표시
            // conversationHistory.push({role: "ai", content: data.message}); // 대화 기록 저장

        } catch (error) {
            console.error('Chat Error:', error);
            addAiMessage(`죄송합니다. 메시지 처리 중 오류가 발생했습니다: ${error.message}`, true);
        } finally {
            typingIndicator.classList.add('hidden'); // 로딩 인디케이터 숨김
            scrollToBottom();
        }
    }

    // 파일 업로드 처리 함수 (이미지/동영상)
    document.getElementById('media-upload').addEventListener('change', function(e) {
        if (e.target.files.length > 0) {
            const file = e.target.files[0];
            fileUploadOptions.classList.add('hidden'); // 파일 선택 옵션 숨김
            addUserMessage(`동영상/이미지 파일(${escapeHtml(file.name)})을 업로드했습니다.`);
            typingIndicator.classList.remove('hidden'); // 로딩 인디케이터 표시
            scrollToBottom();

            const formData = new FormData();
            formData.append('image', file); // 'image'라는 키로 파일 추가

            // 상황 설명을 위한 입력 (선택 사항)
            // const situation = prompt("업로드하는 이미지/동영상에 대한 간단한 상황 설명을 입력해주세요 (예: 친구가 돈을 요구하며 보낸 사진입니다):", "");
            // if (situation && situation.trim() !== "") {
            //     formData.append('situation', situation.trim());
            // }

            fetch('/api/deepfake/analyze_image', { // Flask의 딥페이크 분석 API 경로
                method: 'POST',
                body: formData,
            })
            .then(response => {
                if (!response.ok) {
                    // 서버에서 오류 응답 시, 응답 본문을 JSON으로 파싱 시도 후 오류 메시지 추출
                    return response.json().then(err => {
                        throw new Error(err.error || `서버 분석 중 오류 발생 (${response.status})`);
                    }).catch(() => { // JSON 파싱 실패 시 (응답이 JSON이 아닐 경우)
                        throw new Error(`서버 응답 오류 (${response.status})`);
                    });
                }
                return response.json(); // 정상 응답은 JSON으로 파싱
            })
            .then(data => {
                typingIndicator.classList.add('hidden'); // 로딩 인디케이터 숨김
                console.log('전체 서버 응답 (data):', JSON.stringify(data, null, 2)); // 디버깅용 로그

                if (data.llm_judgment && data.llm_judgment.text) { // llm_judgment와 그 안의 text 필드 확인
                    try {
                        // llm_judgment.text 값을 JSON 객체로 파싱
                        const llmJudgmentObject = JSON.parse(data.llm_judgment.text);
                        console.log('파싱된 LLM 판단 객체:', llmJudgmentObject);

                        const probability = llmJudgmentObject.deepfake_probability || '알 수 없음';
                        const reasoning = llmJudgmentObject.reasoning || '제공되지 않음';
                        const recommendations = llmJudgmentObject.recommendations_for_user || '추가 조언 없음';
                        const confidence = llmJudgmentObject.confidence_score;

                        let resultMessage = `**분석 결과: ${probability}**\n`;
                        if (confidence !== undefined) {
                            resultMessage += `(판단 신뢰도: ${Math.round(confidence * 100)}%)\n\n`;
                        } else {
                            resultMessage += `\n`;
                        }
                        resultMessage += `**판단 근거:**\n${reasoning}\n\n`;
                        resultMessage += `**권장 사항:**\n${recommendations}`;

                        const isAlert = probability.includes("높음") || probability.includes("매우 높음");
                        addAiMessage(resultMessage.trim(), isAlert);

                        // (선택 사항) 얼굴 특징 분석 정보 간략히 표시
                        if (data.feature_analysis && data.feature_analysis.face_detected) {
                            let featureSummary = "부가 정보: ";
                            if (data.feature_analysis.eye_blinking_analysis) {
                                featureSummary += `눈 깜빡임 상태는 '${data.feature_analysis.eye_blinking_analysis.eye_blinking_status}'으로 분석되었습니다. `;
                            }
                            // 다른 특징 정보도 필요시 추가 가능
                            // addAiMessage(featureSummary.trim()); // 별도 메시지로 표시하려면 주석 해제
                        } else if (data.feature_analysis && !data.feature_analysis.face_detected) {
                            addAiMessage("이미지에서 얼굴을 감지하지 못했습니다.");
                        }

                    } catch (parseError) {
                        console.error("LLM 판단 결과 JSON 파싱 오류:", parseError);
                        addAiMessage("AI의 분석 결과를 해석하는 중 오류가 발생했습니다. 응답 형식을 확인해주세요.", true);
                    }
                } else if (data.error) { // 서버가 {"error": "..."} 형태로 응답한 경우
                    addAiMessage(`분석 오류가 발생했습니다: ${data.error}`, true);
                } else { // 그 외 예상치 못한 응답 형식
                    addAiMessage("분석 결과를 받았으나, 예상치 못한 형식입니다. 관리자에게 문의해주세요.");
                }
            })
            .catch(error => { // fetch 자체의 실패 또는 이전 .then()에서 throw된 오류 처리
                typingIndicator.classList.add('hidden');
                console.error('Upload 또는 분석 처리 Error:', error);
                addAiMessage(`파일 업로드 또는 분석 처리 중 오류가 발생했습니다: ${error.message}`, true);
            })
            .finally(() => { // 성공/실패 여부와 관계없이 항상 실행
                 typingIndicator.classList.add('hidden');
                 scrollToBottom();
                 e.target.value = ''; // 파일 선택 초기화 (같은 파일 다시 업로드 가능하도록)
            });
        }
    });

    // 음성 파일 업로드 (데모 - 실제 기능 구현 필요)
    document.getElementById('audio-upload').addEventListener('change', function(e) {
        if (e.target.files.length > 0) {
            const file = e.target.files[0];
            fileUploadOptions.classList.add('hidden');
            addUserMessage(`음성 파일(${escapeHtml(file.name)})을 업로드했습니다.`);
            typingIndicator.classList.remove('hidden');
            scrollToBottom();
            // TODO: 실제 음성 파일 분석 API 호출 로직 구현 필요
            // 이 부분도 위 이미지 업로드와 유사하게 FormData 사용 및 fetch API 호출로 구현
            setTimeout(() => {
                typingIndicator.classList.add('hidden');
                addAiMessage("음성 파일 분석 기능은 현재 개발 중입니다. 곧 만나보실 수 있습니다!");
            }, 1500);
            e.target.value = ''; // 파일 선택 초기화
        }
    });

    // 이벤트 리스너 등록
    sendBtn.addEventListener('click', sendMessage);
    userInput.addEventListener('keypress', function(e) {
        if (e.key === 'Enter') {
            e.preventDefault(); // Enter 키 기본 동작(줄바꿈 등) 방지
            sendMessage();
        }
    });

    uploadBtn.addEventListener('click', function() {
        fileUploadOptions.classList.toggle('hidden');
    });

    closeUploadBtn.addEventListener('click', function() {
        fileUploadOptions.classList.add('hidden');
    });

});