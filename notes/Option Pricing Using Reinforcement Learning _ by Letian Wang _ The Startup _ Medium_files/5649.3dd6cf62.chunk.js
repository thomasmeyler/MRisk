(self.webpackChunklite=self.webpackChunklite||[]).push([[5649],{14524:(e,n,t)=>{"use strict";t.d(n,{xO:()=>c});var i=t(319),a=t.n(i),o=t(89748),r=t(8994),l=t(89080),d={kind:"Document",definitions:[{kind:"FragmentDefinition",name:{kind:"Name",value:"PostPreviewByLine_user"},typeCondition:{kind:"NamedType",name:{kind:"Name",value:"User"}},selectionSet:{kind:"SelectionSet",selections:[{kind:"Field",name:{kind:"Name",value:"id"}},{kind:"Field",name:{kind:"Name",value:"__typename"}},{kind:"FragmentSpread",name:{kind:"Name",value:"CardByline_user"}},{kind:"FragmentSpread",name:{kind:"Name",value:"ExpandablePostByline_user"}}]}}].concat(a()(o.br.definitions),a()(r.A.definitions))},s={kind:"Document",definitions:[{kind:"FragmentDefinition",name:{kind:"Name",value:"PostPreviewByLine_collection"},typeCondition:{kind:"NamedType",name:{kind:"Name",value:"Collection"}},selectionSet:{kind:"SelectionSet",selections:[{kind:"Field",name:{kind:"Name",value:"id"}},{kind:"Field",name:{kind:"Name",value:"__typename"}},{kind:"FragmentSpread",name:{kind:"Name",value:"CardByline_collection"}},{kind:"FragmentSpread",name:{kind:"Name",value:"CollectionLinkWithPopover_collection"}}]}}].concat(a()(o.We.definitions),a()(l.W.definitions))},c={kind:"Document",definitions:[{kind:"FragmentDefinition",name:{kind:"Name",value:"PostPreviewByLine_post"},typeCondition:{kind:"NamedType",name:{kind:"Name",value:"Post"}},selectionSet:{kind:"SelectionSet",selections:[{kind:"Field",name:{kind:"Name",value:"id"}},{kind:"Field",name:{kind:"Name",value:"creator"},selectionSet:{kind:"SelectionSet",selections:[{kind:"FragmentSpread",name:{kind:"Name",value:"PostPreviewByLine_user"}}]}},{kind:"Field",name:{kind:"Name",value:"collection"},selectionSet:{kind:"SelectionSet",selections:[{kind:"FragmentSpread",name:{kind:"Name",value:"PostPreviewByLine_collection"}}]}},{kind:"FragmentSpread",name:{kind:"Name",value:"CardByline_post"}}]}}].concat(a()(d.definitions),a()(s.definitions),a()(o.yu.definitions))}},9842:(e,n,t)=>{"use strict";t.d(n,{G:()=>l});var i=t(67294),a=t(68717),o=t(36764),r=t(77355),l=function(e){var n=e.post,t=e.showAuthor,l=void 0===t||t,d=e.showCollectionName,s=void 0===d||d,c=e.marginBottom,m=n.collection||n.creator,u="User"===(null==m?void 0:m.__typename)&&n.collection?n.collection:n.creator;return l?i.createElement(r.x,{marginBottom:c,display:"flex"},n.creator?i.createElement(o.h,{author:n.creator,collection:s?n.collection:void 0,includeAvatar:!0,includeVerifiedAuthorBadge:!0}):u&&i.createElement(a.u,{publisher:u,publishedAt:void 0,post:n,isOneLine:!0})):null}},3105:(e,n,t)=>{"use strict";t.d(n,{J:()=>l});var i=t(319),a=t.n(i),o=t(69724),r=t(4088),l={kind:"Document",definitions:[{kind:"FragmentDefinition",name:{kind:"Name",value:"PostPreviewContainer_post"},typeCondition:{kind:"NamedType",name:{kind:"Name",value:"Post"}},selectionSet:{kind:"SelectionSet",selections:[{kind:"Field",name:{kind:"Name",value:"id"}},{kind:"Field",name:{kind:"Name",value:"extendedPreviewContent"},selectionSet:{kind:"SelectionSet",selections:[{kind:"Field",name:{kind:"Name",value:"isFullContent"}}]}},{kind:"Field",name:{kind:"Name",value:"visibility"}},{kind:"Field",name:{kind:"Name",value:"pinnedAt"}},{kind:"FragmentSpread",name:{kind:"Name",value:"PostScrollTracker_post"}},{kind:"FragmentSpread",name:{kind:"Name",value:"usePostUrl_post"}}]}}].concat(a()(o.k.definitions),a()(r.u.definitions))}},69935:(e,n,t)=>{"use strict";t.d(n,{V:()=>v});var i=t(59713),a=t.n(i),o=t(67294),r=t(25145),l=t(77355),d=t(66411),s=t(14646),c=t(68821),m=t(13663);function u(e,n){var t=Object.keys(e);if(Object.getOwnPropertySymbols){var i=Object.getOwnPropertySymbols(e);n&&(i=i.filter((function(n){return Object.getOwnPropertyDescriptor(e,n).enumerable}))),t.push.apply(t,i)}return t}function p(e){for(var n=1;n<arguments.length;n++){var t=null!=arguments[n]?arguments[n]:{};n%2?u(Object(t),!0).forEach((function(n){a()(e,n,t[n])})):Object.getOwnPropertyDescriptors?Object.defineProperties(e,Object.getOwnPropertyDescriptors(t)):u(Object(t)).forEach((function(n){Object.defineProperty(e,n,Object.getOwnPropertyDescriptor(t,n))}))}return e}function k(e){var n=e.post,t=e.index,i=e.presentationTrackerReferrerSource,a=e.children,u=e.isFullHeight,k=p(p({},(0,d.Lk)()),{},{index:t}),v=(0,r.D)()(n),f=o.useRef(null);(0,c.V)(f,n);var S=n.pinnedAt,g=n.extendedPreviewContent,h=!(null==g||!g.isFullContent),x=(0,s.I)(),y=u?"100%":void 0;return o.createElement(d.cW,{source:k},o.createElement("article",{className:x({height:y})},o.createElement(l.x,{boxSizing:"content-box",height:y},o.createElement(m.o,{post:p(p({},n),{},{previewContent:{isFullContent:h}}),presentationContext:"POST_PREVIEW",isDisplayingFullPost:h,suppressedEvents:h?"VIEWED":void 0,shouldReportClientViewed:!1,reportClientViewedOnFullPost:!0,postClientViewedContext:2,pinned:!!S,referrerSource:i},o.createElement(l.x,{ref:f,height:y},a({postUrl:v}))))))}var v=(0,o.memo)(k,(function(e,n){return e.post.id===n.post.id&&e.index===n.index}))},8607:(e,n,t)=>{"use strict";t.d(n,{m1:()=>d});var i=t(319),a=t.n(i),o=t(93403),r=t(36579),l=t(10654),d={kind:"Document",definitions:[{kind:"FragmentDefinition",name:{kind:"Name",value:"PostPreviewFooter_post"},typeCondition:{kind:"NamedType",name:{kind:"Name",value:"Post"}},selectionSet:{kind:"SelectionSet",selections:[{kind:"Field",name:{kind:"Name",value:"id"}},{kind:"FragmentSpread",name:{kind:"Name",value:"BookmarkButton_post"}},{kind:"FragmentSpread",name:{kind:"Name",value:"ExpandablePostCardOverflowButton_post"}},{kind:"Field",name:{kind:"Name",value:"allowResponses"}},{kind:"Field",name:{kind:"Name",value:"isLimitedState"}},{kind:"Field",name:{kind:"Name",value:"postResponses"},selectionSet:{kind:"SelectionSet",selections:[{kind:"Field",name:{kind:"Name",value:"count"}}]}},{kind:"FragmentSpread",name:{kind:"Name",value:"MultiVote_post"}},{kind:"Field",name:{kind:"Name",value:"creator"},selectionSet:{kind:"SelectionSet",selections:[{kind:"FragmentSpread",name:{kind:"Name",value:"PostPreviewFooter_user"}}]}},{kind:"Field",name:{kind:"Name",value:"collection"},selectionSet:{kind:"SelectionSet",selections:[{kind:"FragmentSpread",name:{kind:"Name",value:"PostPreviewFooter_collection"}}]}}]}}].concat(a()(o.z.definitions),a()(r.D.definitions),a()(l.x.definitions),a()([{kind:"FragmentDefinition",name:{kind:"Name",value:"PostPreviewFooter_user"},typeCondition:{kind:"NamedType",name:{kind:"Name",value:"User"}},selectionSet:{kind:"SelectionSet",selections:[{kind:"Field",name:{kind:"Name",value:"__typename"}},{kind:"Field",name:{kind:"Name",value:"id"}}]}}]),a()([{kind:"FragmentDefinition",name:{kind:"Name",value:"PostPreviewFooter_collection"},typeCondition:{kind:"NamedType",name:{kind:"Name",value:"Collection"}},selectionSet:{kind:"SelectionSet",selections:[{kind:"Field",name:{kind:"Name",value:"__typename"}},{kind:"Field",name:{kind:"Name",value:"id"}}]}}]))}},55047:(e,n,t)=>{"use strict";t.d(n,{u:()=>g});var i=t(66604),a=t.n(i),o=t(67294),r=t(50455),l=t(78060),d=t(56804),s=t(75761),c=t(6443),m=t(77355),u=t(75221),p=t(78870),k=t(36001);function v(e){var n,t=e.post,i=e.postUrl,a=(0,p.Rk)(i,{responsesOpen:"true",sortBy:u.sV.REVERSE_CHRON});return o.createElement(o.Fragment,null,o.createElement(d.S,{post:t,buttonStyle:"SUBTLE_MARGIN",hasDialog:!0,susiEntry:"clap_footer",buttonColor:"LIGHTER",countScale:"S",shouldHideClapsText:!0,shouldShowResponsiveLabelText:!0}),o.createElement(m.x,{marginLeft:"20px"},o.createElement(s.h,{href:a,responsesCountScale:"S",trackingData:{postId:t.id},responsesCount:(null===(n=t.postResponses)||void 0===n?void 0:n.count)||null,isLimitedState:t.isLimitedState,allowResponses:t.allowResponses,disabledTooltipText:"Responses hidden",countStylesOverride:{marginLeft:"4px",marginTop:"0px"}})))}function f(e){var n,t,i=e.post,a=(0,c.H)().value,d=i.collection||i.creator,s=null!==(n=null==d?void 0:d.__typename)&&void 0!==n?n:null===(t=i.creator)||void 0===t?void 0:t.__typename;return o.createElement(m.x,{alignItems:"center",display:"flex",justifyContent:"flex-end",flexShrink:"0",flexBasis:"0"},o.createElement(r.e,{post:i,susiEntry:"bookmark_preview",targetDistance:15}),s&&a&&o.createElement(m.x,{paddingLeft:"24px"},o.createElement(l.u,{post:i,publisherContext:s})))}var S={S:{flexBasis:"0",maxWidth:"56%"},M:{flexBasis:"0",maxWidth:"56%"},L:{flexBasis:"auto",maxWidth:"unset"}},g=function(e){var n=e.post,t=e.postUrl,i=e.scales,r=(0,k.L)(S,i),l=a()(r,(function(e){return e.flexBasis})),d=a()(r,(function(e){return e.maxWidth}));return o.createElement(m.x,{display:"flex",justifyContent:"space-between"},o.createElement(m.x,{alignItems:"center",display:"flex",flexGrow:"1",flexShrink:"0",padding:"0",flexBasis:l,maxWidth:d},o.createElement(v,{post:n,postUrl:t})),o.createElement(f,{post:n}))}},36001:(e,n,t)=>{"use strict";t.d(n,{Il:()=>o,L:()=>r,n0:()=>l});var i=t(67294),a=t(21755);function o(e){return i.useMemo((function(){return Object.keys(e).reduce((function(n,t){var i=e[t];return n[i]&&n[i].push(t),n}),{S:[],M:[],L:[]})}),[e])}function r(e,n){return{xs:e[n.xs],sm:e[n.sm],md:e[n.md],lg:e[n.lg],xl:e[n.xl]}}function l(e){var n={};for(var t in a.j)n[t]=e;return n}},35558:(e,n,t)=>{"use strict";t.d(n,{w:()=>m});var i=t(319),a=t.n(i),o={kind:"Document",definitions:[{kind:"FragmentDefinition",name:{kind:"Name",value:"StreamPostPreviewImage_post"},typeCondition:{kind:"NamedType",name:{kind:"Name",value:"Post"}},selectionSet:{kind:"SelectionSet",selections:[{kind:"Field",name:{kind:"Name",value:"title"}},{kind:"Field",name:{kind:"Name",value:"previewImage"},selectionSet:{kind:"SelectionSet",selections:[{kind:"FragmentSpread",name:{kind:"Name",value:"StreamPostPreviewImage_imageMetadata"}}]}}]}}].concat(a()([{kind:"FragmentDefinition",name:{kind:"Name",value:"StreamPostPreviewImage_imageMetadata"},typeCondition:{kind:"NamedType",name:{kind:"Name",value:"ImageMetadata"}},selectionSet:{kind:"SelectionSet",selections:[{kind:"Field",name:{kind:"Name",value:"id"}},{kind:"Field",name:{kind:"Name",value:"focusPercentX"}},{kind:"Field",name:{kind:"Name",value:"focusPercentY"}},{kind:"Field",name:{kind:"Name",value:"alt"}}]}}]))},r=t(8607),l=t(14524),d=t(63009),s={kind:"Document",definitions:[{kind:"FragmentDefinition",name:{kind:"Name",value:"StreamPostPreviewContent_post"},typeCondition:{kind:"NamedType",name:{kind:"Name",value:"Post"}},selectionSet:{kind:"SelectionSet",selections:[{kind:"Field",name:{kind:"Name",value:"id"}},{kind:"Field",name:{kind:"Name",value:"title"}},{kind:"Field",name:{kind:"Name",value:"previewImage"},selectionSet:{kind:"SelectionSet",selections:[{kind:"Field",name:{kind:"Name",value:"id"}}]}},{kind:"Field",name:{kind:"Name",value:"extendedPreviewContent"},selectionSet:{kind:"SelectionSet",selections:[{kind:"Field",name:{kind:"Name",value:"subtitle"}}]}},{kind:"FragmentSpread",name:{kind:"Name",value:"StreamPostPreviewImage_post"}},{kind:"FragmentSpread",name:{kind:"Name",value:"PostPreviewFooter_post"}},{kind:"FragmentSpread",name:{kind:"Name",value:"PostPreviewByLine_post"}},{kind:"FragmentSpread",name:{kind:"Name",value:"PostPreviewInformation_post"}}]}}].concat(a()(o.definitions),a()(r.m1.definitions),a()(l.xO.definitions),a()(d.u.definitions))},c=t(3105),m={kind:"Document",definitions:[{kind:"FragmentDefinition",name:{kind:"Name",value:"StreamPostPreview_post"},typeCondition:{kind:"NamedType",name:{kind:"Name",value:"Post"}},selectionSet:{kind:"SelectionSet",selections:[{kind:"Field",name:{kind:"Name",value:"id"}},{kind:"FragmentSpread",name:{kind:"Name",value:"StreamPostPreviewContent_post"}},{kind:"FragmentSpread",name:{kind:"Name",value:"PostPreviewContainer_post"}}]}}].concat(a()(s.definitions),a()(c.J.definitions))}},26058:(e,n,t)=>{"use strict";t.d(n,{j:()=>C});var i=t(67294),a=t(66604),o=t.n(a),r=t(77355),l=t(93310),d=t(18634),s=t(52069),c=t(90586),m=t(87691),u=t(97480),p=t(31889),k=t(36001),v={S:"24px",M:"32px",L:"32px"};function f(e){var n=e.scales,t=e.dividerColor,a=(0,p.F)(),o=(0,k.L)(v,n);return i.createElement(u.E,{marginTop:o,borderColor:null!=t?t:a.colorTokens.border.neutral.primary.base})}var S=t(4381),g=t(46696),h=function(e){return{backgroundColor:e.colorTokens.background.neutral.tertiary.base,borderRadius:"2px"}},x={S:{height:53,width:80},M:{height:53,width:80},L:{height:107,width:160}},y=function(e){var n=e.post,t=e.postUrl,a=e.scales,o=(0,k.Il)(a),r=n.previewImage,s=n.title,c=null==r?void 0:r.id;if(!c)return null;var m=r.alt||s||"";return i.createElement(l.r,{href:t,"aria-label":m||"Post Preview Image"},Object.keys(o).map((function(e){var n=o[e];return n.length?i.createElement(d.y,{xs:n.includes("xs"),sm:n.includes("sm"),md:n.includes("md"),lg:n.includes("lg"),xl:n.includes("xl"),key:"stream-image-".concat(e)},i.createElement(S.UV,{alt:m,miroId:c,width:x[e].width,height:x[e].height,strategy:g._S.Crop,focusPercentX:r.focusPercentX,focusPercentY:r.focusPercentY,rules:h})):null})))},N=t(9842),E=t(55047),w=t(63254),F={S:{footerLocation:"bottom",showSubtitle:!1,subElementSpacing:"12px",titleScale:"XS",titleClamp:2,imageMarginLeft:"24px"},M:{footerLocation:"bottom",showSubtitle:!0,subElementSpacing:"16px",titleScale:"S",titleClamp:2,imageMarginLeft:"24px"},L:{footerLocation:"content",showSubtitle:!0,subElementSpacing:"20px",titleScale:"M",titleClamp:3,imageMarginLeft:"56px"}},b=function(e){var n,t,a,u,p=e.post,v=e.postUrl,S=e.scales,g=e.showDivider,h=e.showCollectionName,x=e.showAuthor,b=e.dividerColor,P=p.title,C=null==p||null===(n=p.extendedPreviewContent)||void 0===n?void 0:n.subtitle,_=null===(t=p.previewImage)||void 0===t?void 0:t.id,L=(0,k.L)(F,S),O=o()(L,(function(e){return e.subElementSpacing})),D=o()(L,(function(e){return e.titleScale})),T=o()(L,(function(e){return e.titleClamp})),R=o()(L,(function(e){return e.imageMarginLeft}));return i.createElement(r.x,null,i.createElement(N.G,{post:p,marginBottom:O,showCollectionName:h,showAuthor:x}),i.createElement(r.x,{display:"flex"},i.createElement(r.x,{flexGrow:"1",flexShrink:"1",wordBreak:"break-word"},i.createElement(l.r,{href:v},P&&i.createElement(s.Dx,{scale:D,clamp:T},P),C&&i.createElement(d.y,{xs:L.xs.showSubtitle,sm:L.sm.showSubtitle,md:L.md.showSubtitle,lg:L.lg.showSubtitle,xl:L.xl.showSubtitle,paddingTop:"8px"},i.createElement(c.QE,{scale:"S",clamp:(a=(null==P?void 0:P.length)||0,u=!!_,u?a>104?1:a>52?2:3:a>140?1:a>70?2:3)},C))),i.createElement(l.r,{href:v},i.createElement(m.F,{scale:"S",tag:"span"},i.createElement(r.x,{display:"flex",alignItems:"center",paddingTop:O},i.createElement(w.O,{post:p})))),i.createElement(d.y,{xs:"content"===L.xs.footerLocation,sm:"content"===L.sm.footerLocation,md:"content"===L.md.footerLocation,lg:"content"===L.lg.footerLocation,xl:"content"===L.xl.footerLocation,paddingTop:"24px"},i.createElement(E.u,{post:p,postUrl:v,scales:S}))),!!_&&i.createElement(r.x,{marginLeft:R},i.createElement(y,{post:p,postUrl:v,scales:S}))),i.createElement(d.y,{xs:"bottom"===L.xs.footerLocation,sm:"bottom"===L.sm.footerLocation,md:"bottom"===L.md.footerLocation,lg:"bottom"===L.lg.footerLocation,xl:"bottom"===L.xl.footerLocation,paddingTop:"16px"},i.createElement(E.u,{post:p,postUrl:v,scales:S})),g&&i.createElement(f,{scales:S,dividerColor:b}))},P=t(69935);function C(e){var n=e.post,t=e.index,a=e.presentationTrackerReferrerSource,o=e.scale,r=e.lastIndex,l=e.showCollectionName,d=e.showAuthor,s=e.dividerColor,c="string"==typeof o?(0,k.n0)(o):o;return i.createElement(P.V,{post:n,index:t,presentationTrackerReferrerSource:a},(function(e){var a=e.postUrl;return i.createElement(b,{post:n,postUrl:a,scales:c,showDivider:t!==r,showCollectionName:l,showAuthor:d,dividerColor:s})}))}},36764:(e,n,t)=>{"use strict";t.d(n,{h:()=>f});var i=t(67294),a=t(84739),o=t(65968),r=t(64238),l=t(26700),d=t(17193),s=t(28695),c=t(77355),m=t(69992),u=t(93310),p=t(30020),k=t(87691),v=function(e){var n=e.author;return(0,r.o)(n)?i.createElement(c.x,{marginLeft:"2px",marginTop:"2px"},i.createElement(o._,{size:"S"})):null},f=function(e){var n,t=(0,a.I)()(e.author),o=(0,i.useCallback)((function(){return e.author&&i.createElement(s.K,{user:e.author})}),[e.author]);return i.createElement(i.Fragment,null,e.includeAvatar&&i.createElement(c.x,{marginRight:"8px"},i.createElement(m.$,{targetDistance:15,mouseLeaveDelay:100,mouseEnterDelay:p.w,placement:"top",role:"tooltip",popoverRenderFn:o},i.createElement(u.r,{href:t},i.createElement(d.Yt,{scale:"XXXS",user:e.author,showHoverState:!0})))),i.createElement(c.x,{paddingRight:"4px"},i.createElement(m.$,{targetDistance:15,mouseLeaveDelay:100,mouseEnterDelay:p.w,placement:"top",role:"tooltip",popoverRenderFn:o},i.createElement(u.r,{href:t,linkStyle:"SUBTLE",inline:!0,display:"flex",rules:{alignItems:"center"}},i.createElement(k.F,{scale:"S",color:"DARKER",clamp:1},e.author.name),e.includeVerifiedAuthorBadge&&i.createElement(v,{author:e.author})))),(null===(n=e.collection)||void 0===n?void 0:n.name)&&i.createElement(i.Fragment,null,i.createElement(c.x,{paddingRight:"4px"},i.createElement(k.F,{scale:"S",color:"LIGHTER"},"in")),i.createElement(c.x,null,i.createElement(l.q,{collection:e.collection,clamp:1,popoverPlacement:"top",scale:"S"}))))}},75761:(e,n,t)=>{"use strict";t.d(n,{h:()=>F});var i=t(59713),a=t.n(i),o=t(67294),r=t(93310),l=t(30020),d=t(87691),s=t(18627),c=t(66411),m=t(14646);function u(){return(u=Object.assign||function(e){for(var n=1;n<arguments.length;n++){var t=arguments[n];for(var i in t)Object.prototype.hasOwnProperty.call(t,i)&&(e[i]=t[i])}return e}).apply(this,arguments)}var p=o.createElement("path",{d:"M18 16.8a7.14 7.14 0 0 0 2.24-5.32c0-4.12-3.53-7.48-8.05-7.48C7.67 4 4 7.36 4 11.48c0 4.13 3.67 7.48 8.2 7.48a8.9 8.9 0 0 0 2.38-.32c.23.2.48.39.75.56 1.06.69 2.2 1.04 3.4 1.04.22 0 .4-.11.48-.29a.5.5 0 0 0-.04-.52 6.4 6.4 0 0 1-1.16-2.65v.02zm-3.12 1.06l-.06-.22-.32.1a8 8 0 0 1-2.3.33c-4.03 0-7.3-2.96-7.3-6.59S8.17 4.9 12.2 4.9c4 0 7.1 2.96 7.1 6.6 0 1.8-.6 3.47-2.02 4.72l-.2.16v.26l.02.3a6.74 6.74 0 0 0 .88 2.4 5.27 5.27 0 0 1-2.17-.86c-.28-.17-.72-.38-.94-.59l.01-.02z"});const k=function(e){return o.createElement("svg",u({width:24,height:24,viewBox:"0 0 24 24"},e),p)};function v(){return(v=Object.assign||function(e){for(var n=1;n<arguments.length;n++){var t=arguments[n];for(var i in t)Object.prototype.hasOwnProperty.call(t,i)&&(e[i]=t[i])}return e}).apply(this,arguments)}var f=o.createElement("path",{fillRule:"evenodd",clipRule:"evenodd",d:"M18.47 20.27a6.08 6.08 0 0 1-4.06-1.55c-.74.2-1.51.3-2.29.3-4.48 0-8.12-3.35-8.12-7.48 0-4.15 3.64-7.5 8.12-7.5 4.48 0 8.12 3.35 8.12 7.48 0 1.98-.81 3.83-2.3 5.23.02.17.05.34.1.53.2.66.52 1.33 1 1.96a.66.66 0 0 1-.53 1.04h-.04z"});const S=function(e){return o.createElement("svg",v({width:24,height:24,viewBox:"0 0 24 24"},e),f)};function g(e,n){var t=Object.keys(e);if(Object.getOwnPropertySymbols){var i=Object.getOwnPropertySymbols(e);n&&(i=i.filter((function(n){return Object.getOwnPropertyDescriptor(e,n).enumerable}))),t.push.apply(t,i)}return t}function h(e){for(var n=1;n<arguments.length;n++){var t=null!=arguments[n]?arguments[n]:{};n%2?g(Object(t),!0).forEach((function(n){a()(e,n,t[n])})):Object.getOwnPropertyDescriptors?Object.defineProperties(e,Object.getOwnPropertyDescriptors(t)):g(Object(t)).forEach((function(n){Object.defineProperty(e,n,Object.getOwnPropertyDescriptor(t,n))}))}return e}var x=function(e,n){return"LIGHTER"===n?e.colorTokens.foreground.neutral.primary.hover:e.colorTokens.foreground.neutral.secondary.base},y=function(e,n){return function(t){return{cursor:n?"not-allowed":"pointer",border:0,opacity:n?.25:1,padding:"4px 0",display:"flex",alignItems:"center",fill:"LIGHTER"===e?t.colorTokens.foreground.neutral.secondary.base:t.baseColor.fill.light,":hover":n?void 0:{fill:x(t,e),"& p":{color:x(t,e)}}}}},N=function(e){return{cursor:"not-allowed",fill:e.colorTokens.foreground.neutral.secondary.base,opacity:.25}},E=function(e){var n=e.allowResponses,t=e.handleClick,i=e.href,a=e.children,l=e.responsesCountColor,d=e.isLimitedState,s=(0,m.I)(),c=n?"responses":"responses hidden";return i?o.createElement(r.r,{onClick:t,href:i,rules:y(l,d),"aria-label":c,disabled:!n},a):o.createElement("button",{onClick:t,className:s(y(l,d)),"aria-label":c,disabled:!n},a)},w=function(e){var n=e.allowResponses,t=e.iconStylesOverride,i=(0,m.I)();return n?o.createElement(k,{className:i([t])}):o.createElement(S,{className:i([N])})},F=function(e){var n=e.allowResponses,t=e.responsesCount,i=e.handleClick,a=e.trackingData,r=e.isLimitedState,u=e.iconStylesOverride,p=e.countStylesOverride,k=e.responsesCountColor,v=void 0===k?"LIGHTER":k,f=e.disabledTooltipText,S=void 0===f?"":f,g=e.responsesCountScale,x=void 0===g?"M":g,y=e.href,N=(0,m.I)(),F=(0,s.Av)(),b=(0,c.pK)(),P=n&&t,C=(0,o.useCallback)((function(e){null==i||i(e),F.event("responses.viewAllClicked",h(h({},a),{},{source:b}))}),[F,i,a,b]);return o.createElement(l._,{tooltipText:r||!n?S:"Respond",targetDistance:15},o.createElement(E,{handleClick:r||!n?void 0:C,responsesCountColor:v,allowResponses:n,isLimitedState:r,href:y},o.createElement(w,{allowResponses:n,iconStylesOverride:u}),!!P&&o.createElement(d.F,{scale:x,color:v},o.createElement("span",{className:"pw-responses-count ".concat(N([p]))},t))))}},27048:(e,n,t)=>{"use strict";t.d(n,{W:()=>r});var i=t(319),a=t.n(i),o=t(68216),r={kind:"Document",definitions:[{kind:"FragmentDefinition",name:{kind:"Name",value:"UserAvatar_user"},typeCondition:{kind:"NamedType",name:{kind:"Name",value:"User"}},selectionSet:{kind:"SelectionSet",selections:[{kind:"Field",name:{kind:"Name",value:"__typename"}},{kind:"Field",name:{kind:"Name",value:"id"}},{kind:"Field",name:{kind:"Name",value:"imageId"}},{kind:"Field",name:{kind:"Name",value:"mediumMemberAt"}},{kind:"Field",name:{kind:"Name",value:"name"}},{kind:"Field",name:{kind:"Name",value:"username"}},{kind:"FragmentSpread",name:{kind:"Name",value:"userUrl_user"}}]}}].concat(a()(o.$m.definitions))}}}]);
//# sourceMappingURL=https://stats.medium.build/lite/sourcemaps/5649.3dd6cf62.chunk.js.map