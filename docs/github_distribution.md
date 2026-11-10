# GitHub 코드와 학습 산출물 배포

코드 저장소에는 환경·학습기·설정·테스트·모델 자산·작은 평가 요약을 보관합니다.
체크포인트·추론 가중치·영상·캐시·개인 실행 출력은 별도 배포 파일로 관리합니다.

## 8× 배포 파일 다운로드

[8× MLP Release](https://github.com/redEddie/piper-sim-lab/releases/tag/show76-v1-8x-20261023-r2)에서
아래 두 파일을 받습니다. 저장소가 비공개이므로 접근 가능한 GitHub 계정으로 로그인해야 합니다.

검증된 8× 원본을 수정하지 않은 파일:

```text
thor-show76-v1-8x-20261023-r2.tar.gz
thor-show76-v1-8x-20261023-r2.tar.gz.sha256
```

아카이브 크기는 33,829,353 bytes입니다. 원본 checkpoint·normalizer·NumPy export·평가 기록·
런타임·checksum과 학습 당시 코드 Git bundle을 포함합니다.
실행 호스트 경로는 provenance로 보존되어 있으며 이식 후 실제 파일은 추출한 release 경로를 사용합니다.
GitHub Release에서 코드 clone과 별도로 받을 수 있습니다.

GitHub CLI를 사용하는 경우 저장소 루트에서 다음을 실행합니다.

```bash
gh release download show76-v1-8x-20261023-r2 \
  --repo redEddie/piper-sim-lab \
  --pattern 'thor-show76-v1-8x-20261023-r2.tar.gz*' --dir releases
python scripts/deployment/verify_release.py \
  releases/thor-show76-v1-8x-20261023-r2.tar.gz \
  --extract-to releases/show76-v1-8x
PYTHONPATH=runtime/src python -m piper_policy_runtime.check \
  --policy releases/show76-v1-8x/policies/show76-v1-mlp
```

브라우저에서 파일을 받았다면 아래처럼 받은 경로를 지정합니다.

```bash
python scripts/deployment/verify_release.py /path/to/thor-show76-v1-8x-20261023-r2.tar.gz --extract-to /path/to/new-release
PYTHONPATH=runtime/src python -m piper_policy_runtime.check --policy /path/to/new-release/policies/show76-v1-mlp
```

배포 파일과 .sha256을 같은 디렉터리에 받습니다. 검증기는 기존 목적지를 덮어쓰지 않습니다.
아카이브 SHA-256과 내부 파일 manifest를 검사합니다. 위 명령은 정책 연산을 확인하며
하드웨어에 명령을 전송하지 않습니다.

## 보존할 버전 정보

- 학습 코드: `b40fa4df0c8ef301421edab65250f9e3ed70037a`.
- 기존 `show76-v1-20261023-r1` tag와 3× 정책은 그대로 보존합니다.
- 8× 배포 ID: `thor-show76-v1-8x-20261023-r2`.
- GitHub Release tag: `show76-v1-8x-20261023-r2`. 문서·시연을 포함한 배포용 코드 버전을 가리키며
  학습 당시 commit은 위의 `b40fa4d`입니다.
- 결과·제약: [v1 릴리스 문서](show76_v1_release.md).
- 자산 라이선스: [ASSETS.md](../ASSETS.md)와 원본 고지.

프로젝트 코드의 라이선스는 아직 지정하지 않았습니다. 외부 자산의 MIT 고지를 프로젝트 전체의
라이선스로 확대해 표시하지 않습니다.

이 배포 파일은 선택한 8× 원본 체크포인트를 포함합니다. 과거 2×·3× 체크포인트는 별도 보관하며
이 파일에는 포함하지 않습니다. 아카이브 안의 문서는 생성 당시 기록이며 현재 안내는 GitHub 문서를 따릅니다.
